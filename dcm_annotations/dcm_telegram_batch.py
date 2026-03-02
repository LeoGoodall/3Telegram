#!/usr/bin/env python3
"""
Full pipeline for DCM annotation of Telegram extremist channel data.
Generates batch files, submits to OpenAI's GPT-4o-mini batch API,
downloads results, and creates analysis-ready CSV with message metadata.

Usage:
  python dcm_telegram_batch.py                # Generate + submit (interactive)
  python dcm_telegram_batch.py --submit-only  # Regenerate with current prompts + submit
  python dcm_telegram_batch.py --status       # Check status + resubmit failed
  python dcm_telegram_batch.py --retrieve     # Download result files only
  python dcm_telegram_batch.py --parse        # Parse downloaded results into analysis CSV

DCM variables annotated:
- Identity Fusion
- Violent Language / Violence Condoning
- Perceived Existential Threat
- Outgroup Hostility/Othering
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
import httpx

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=httpx.Timeout(120.0, connect=30.0)
)

# Configuration
BASE_DIR = "/Volumes/One Touch/terrorgram"
JSON_DIR = os.path.join(BASE_DIR, "json_output")
CATEGORIES_CSV = os.path.join(BASE_DIR, "channel_categories.csv")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "dcm_annotations")

MODEL = "gpt-4o-mini"
MAX_REQUESTS_PER_BATCH = 10_000
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200MB

# Categories to include (excluding proud-boys per user specification)
INCLUDED_CATEGORIES = {"764", "terrorgram", "terrorgram-affiliated", "white-supremacist"}
CATEGORY_MAP = {"terrorgram-affiliated": "terrorgram"}  # Merge affiliated into terrorgram

# DCM feature names and their corresponding prompt files
FEATURES = {
    "identityfusion": "identityfusion.txt",
    "violentlang": "violentlanguage.txt",
    "threat": "threat.txt",
    "outgroup_othering": "othering.txt"
}


def clean_text(text):
    """
    Clean text by removing HTML tags, entities, and URLs.
    Only removes actual HTML syntax, not words that might coincidentally match tag names.
    """
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else ""

    # Remove HTML tags (including self-closing like <br/>, <br />, and tags with attributes like <a href="...">)
    # This handles: <tag>, <tag/>, <tag />, <tag attr="value">, </tag>
    text = re.sub(r'<[^>]+/?>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Catch any remaining tags

    # Remove HTML entities (named and numeric)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # &nbsp; &amp; &lt; &gt; &quot; etc.
    text = re.sub(r'&#\d+;', ' ', text)  # &#123; (decimal)
    text = re.sub(r'&#x[a-fA-F0-9]+;', ' ', text)  # &#xAB; (hex)

    # Remove orphaned HTML attributes (attr="value" patterns outside of tags)
    text = re.sub(r'\bhref\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bclass\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstyle\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsrc\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\balt\s*=\s*["\'][^"\']*["\']', ' ', text, flags=re.IGNORECASE)

    # Remove URLs (http, https, www, t.me)
    text = re.sub(r'https?://\S+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'www\.\S+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r't\.me/\S+', ' ', text, flags=re.IGNORECASE)

    # Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_prompts():
    """Load all DCM feature prompts from files."""
    prompts = {}
    for feature, prompt_file in FEATURES.items():
        prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts[feature] = f.read().strip()
    return prompts


def channel_to_filename(name: str) -> str:
    """
    Convert a channel display name to the corresponding JSON filename stem.
    Matches the logic from step1_topic_modelling.py.
    """
    name = str(name).strip()
    parts = []
    for ch in name:
        if ch.isspace() or ch in {"-", "–"}:
            parts.append("_")
        elif ch.isalnum() or ch == "_":
            parts.append(ch)
        else:
            continue

    slug = "".join(parts)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def load_categories(csv_path):
    """
    Load channel categories from CSV.
    Returns dict: {channel_name: category}
    """
    import csv
    channels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel = row.get('channel', '').strip()
            category = row.get('category', '').strip()
            if channel and category and category in INCLUDED_CATEGORIES:
                # Apply category mapping (e.g., terrorgram-affiliated -> terrorgram)
                mapped_category = CATEGORY_MAP.get(category, category)
                channels[channel] = mapped_category
    return channels


def iter_messages(json_path):
    """
    Iterate through messages in a Telegram JSON export file.
    Yields tuples of (message_id, text).
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error loading {json_path}: {e}")
        return

    messages = data.get('messages', [])
    if not isinstance(messages, list):
        return

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        message_id = msg.get('message_id')
        if not message_id:
            continue

        # Extract text from content.text or content.text_plain
        content = msg.get('content', {})
        if isinstance(content, dict):
            text = content.get('text') or content.get('text_plain')
        else:
            text = None

        # Handle text as list (mixed content)
        if isinstance(text, list):
            parts = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    t = part.get('text')
                    if isinstance(t, str):
                        parts.append(t)
            text = " ".join(parts)

        if text and isinstance(text, str) and text.strip():
            yield str(message_id), text


def create_task(custom_id, text, prompt, model):
    """Create a single batch task for OpenAI API."""
    text_no_links = clean_text(text)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": text_no_links
                }
            ]
        }
    }


class BatchWriter:
    """Handles writing batch files with size and count limits."""

    def __init__(self, output_dir, max_requests, max_size):
        self.output_dir = output_dir
        self.max_requests = max_requests
        self.max_size = max_size
        self.batch_idx = 1
        self.batch_tracking = []

        # Current batch state
        self.current_file = None
        self.current_path = None
        self.current_size = 0
        self.current_count = 0

        os.makedirs(output_dir, exist_ok=True)
        self._start_new_batch()

    def _start_new_batch(self):
        """Start a new batch file."""
        if self.current_file is not None:
            self.current_file.close()

        self.current_path = os.path.join(
            self.output_dir,
            f"batch_telegram_dcm_{self.batch_idx:04d}.jsonl"
        )
        self.current_file = open(self.current_path, 'w', encoding='utf-8')
        self.current_size = 0
        self.current_count = 0

    def _close_current_batch(self):
        """Close and record the current batch."""
        if self.current_count > 0:
            self.current_file.close()
            self.batch_tracking.append({
                "batch_number": self.batch_idx,
                "file_name": self.current_path,
                "file_id": None,
                "batch_job_id": None,
                "task_count": self.current_count,
                "file_size_mb": round(self.current_size / (1024 * 1024), 2)
            })
            print(f"  Created batch {self.batch_idx}: {self.current_count:,} tasks, "
                  f"{self.current_size / 1024 / 1024:.1f}MB")
            self.batch_idx += 1
            return True
        return False

    def write_task(self, task):
        """Write a task to the current batch, rotating if necessary."""
        line = json.dumps(task, ensure_ascii=False) + "\n"
        encoded = line.encode('utf-8')
        line_size = len(encoded)

        # Check if we need to start a new batch
        if (self.current_size + line_size > self.max_size or
                self.current_count >= self.max_requests):
            self._close_current_batch()
            self._start_new_batch()

        self.current_file.write(line)
        self.current_size += line_size
        self.current_count += 1

    def finalise(self):
        """Finalise all batches and return tracking info."""
        self._close_current_batch()
        return self.batch_tracking


def process_all_data(json_dir, categories, output_dir, prompts):
    """Process all JSON files and create batch tasks."""
    print(f"\n{'=' * 60}")
    print("Processing Telegram Extremist Channel Data")
    print(f"{'=' * 60}\n")

    batch_writer = BatchWriter(output_dir, MAX_REQUESTS_PER_BATCH, MAX_FILE_SIZE_BYTES)

    total_messages = 0
    total_tasks = 0
    total_skipped_dupes = 0
    processed_channels = 0
    skipped_channels = 0

    # Track by category
    category_stats = {}

    # Track processed stems to avoid reprocessing when multiple channel names
    # map to the same filename (e.g. Unicode/emoji variants of the same channel)
    processed_stems = set()
    # Track all seen base IDs (category_channel_messageid) to catch duplicates
    seen_base_ids = set()

    for channel_name, category in sorted(categories.items()):
        # Skip macOS hidden files
        if channel_name.startswith('._'):
            continue

        # Convert channel name to filename
        filename_stem = channel_to_filename(channel_name)
        if not filename_stem:
            skipped_channels += 1
            continue

        # Skip if this stem+category was already processed (stem collision)
        stem_key = (category, filename_stem)
        if stem_key in processed_stems:
            print(f"  Skipping duplicate stem: '{channel_name}' -> {filename_stem} (already processed)")
            skipped_channels += 1
            continue
        processed_stems.add(stem_key)

        json_path = os.path.join(json_dir, f"{filename_stem}.json")
        if not os.path.exists(json_path):
            print(f"  Warning: JSON file not found for '{channel_name}' -> {filename_stem}.json")
            skipped_channels += 1
            continue

        print(f"Processing: {channel_name} ({category})")

        # Initialize category stats
        if category not in category_stats:
            category_stats[category] = {"channels": 0, "messages": 0, "tasks": 0}

        channel_message_count = 0
        channel_task_count = 0
        channel_dupes = 0

        for message_id, text in iter_messages(json_path):
            if not text or not text.strip():
                continue

            # Skip very short messages (less than 3 words after cleaning)
            cleaned = clean_text(text)
            if len(cleaned.split()) < 3:
                continue

            # Check for duplicate message_id within this channel (some JSON files
            # contain merged exports with overlapping message_id ranges)
            base_id = f"{category}_{filename_stem}_{message_id}"
            if base_id in seen_base_ids:
                channel_dupes += 1
                total_skipped_dupes += 1
                continue
            seen_base_ids.add(base_id)

            total_messages += 1
            channel_message_count += 1

            # Create a task for each DCM feature
            for feature, prompt in prompts.items():
                # Custom ID format: category_channel_messageid_feature
                # Use filename_stem for channel to avoid special characters
                custom_id = f"{category}_{filename_stem}_{message_id}_{feature}"

                task = create_task(custom_id, text, prompt, MODEL)
                batch_writer.write_task(task)

                total_tasks += 1
                channel_task_count += 1

        processed_channels += 1
        category_stats[category]["channels"] += 1
        category_stats[category]["messages"] += channel_message_count
        category_stats[category]["tasks"] += channel_task_count

        if channel_message_count > 0:
            msg = f"  -> {channel_message_count:,} messages, {channel_task_count:,} tasks"
            if channel_dupes > 0:
                msg += f" ({channel_dupes:,} duplicate message_ids skipped)"
            print(msg)

    batch_tracking = batch_writer.finalise()

    print(f"\n{'=' * 60}")
    print("Processing Complete")
    print(f"{'=' * 60}")
    print(f"\nChannels processed: {processed_channels}")
    print(f"Channels skipped (not found): {skipped_channels}")
    print(f"\nCategory breakdown:")
    for cat, stats in sorted(category_stats.items()):
        print(f"  {cat}: {stats['channels']} channels, {stats['messages']:,} messages, {stats['tasks']:,} tasks")
    print(f"\nTotal messages: {total_messages:,}")
    print(f"Total tasks: {total_tasks:,} ({total_messages:,} messages x {len(prompts)} features)")
    if total_skipped_dupes > 0:
        print(f"Duplicate message_ids skipped: {total_skipped_dupes:,}")
    print(f"Batch files created: {len(batch_tracking)}")

    return batch_tracking


def submit_batches(batch_tracking, tracking_file):
    """Upload batch files and create batch jobs."""
    print(f"\n{'=' * 60}")
    print("Submitting Batch Jobs to OpenAI")
    print(f"{'=' * 60}\n")

    successful = 0
    failed = 0

    for i, batch_info in enumerate(batch_tracking):
        batch_num = batch_info['batch_number']
        file_name = batch_info['file_name']
        task_count = batch_info['task_count']

        print(f"Submitting batch {i + 1}/{len(batch_tracking)}: "
              f"{os.path.basename(file_name)} ({task_count:,} tasks)")

        try:
            # Upload file to OpenAI
            with open(file_name, 'rb') as f:
                batch_file = client.files.create(
                    file=f,
                    purpose='batch'
                )

            # Create batch job
            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint='/v1/chat/completions',
                completion_window="24h"
            )

            # Update batch tracking with file and job IDs
            batch_info['file_id'] = batch_file.id
            batch_info['batch_job_id'] = batch_job.id
            batch_info['status'] = 'submitted'
            batch_info['submitted_at'] = datetime.now().isoformat()

            print(f"  -> Submitted: {batch_job.id}")
            successful += 1

        except Exception as e:
            print(f"  -> Error: {e}")
            batch_info['error'] = str(e)
            batch_info['status'] = 'failed'
            failed += 1

        # Save tracking after each submission (in case of interruption)
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("Submission Complete")
    print(f"{'=' * 60}")
    print(f"Successfully submitted: {successful}/{len(batch_tracking)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(batch_tracking)}")

    return batch_tracking


def main():
    """Main entry point."""
    print(f"\n{'=' * 60}")
    print("Telegram DCM Annotation Batch Pipeline")
    print(f"{'=' * 60}")
    print(f"JSON Directory: {JSON_DIR}")
    print(f"Categories CSV: {CATEGORIES_CSV}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print(f"Max requests per batch: {MAX_REQUESTS_PER_BATCH:,}")
    print(f"Max file size: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB")
    print(f"DCM Features: {', '.join(FEATURES.keys())}")

    # Create output directories
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    batch_files_dir = os.path.join(BASE_OUTPUT_DIR, "batch_files")
    os.makedirs(batch_files_dir, exist_ok=True)

    tracking_file = os.path.join(BASE_OUTPUT_DIR, "batch_tracking.json")

    # Step 1: Load prompts
    print("\nLoading DCM prompts...")
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} feature prompts: {', '.join(prompts.keys())}")

    # Step 2: Load channel categories
    print("\nLoading channel categories...")
    categories = load_categories(CATEGORIES_CSV)
    print(f"Found {len(categories)} channels in included categories")

    # Show category distribution
    cat_counts = {}
    for cat in categories.values():
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} channels")

    # Step 3: Process all data and create batch files
    batch_tracking = process_all_data(JSON_DIR, categories, batch_files_dir, prompts)

    if not batch_tracking:
        print("\nNo batch files created. Exiting.")
        return

    # Step 4: Save initial tracking (before submission)
    print(f"\nSaving batch tracking to {tracking_file}...")
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

    # Step 5: Submit batches
    print("\nReady to submit batches to OpenAI.")
    user_input = input("Proceed with submission? [y/N]: ").strip().lower()

    if user_input == 'y':
        batch_tracking = submit_batches(batch_tracking, tracking_file)

        # Final save
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

        print(f"\nPipeline complete!")
        print(f"Track progress with batch IDs saved in: {tracking_file}")
    else:
        print("\nSubmission skipped. Batch files are ready in:")
        print(f"  {batch_files_dir}")
        print(f"Tracking file saved to: {tracking_file}")
        print("\nRun with --submit-only flag to submit existing batches.")


def submit_only():
    """Regenerate batch files with current prompts and submit (non-interactive)."""
    print(f"\n{'=' * 60}")
    print("Telegram DCM Annotation Batch Pipeline (submit-only)")
    print(f"{'=' * 60}")
    print(f"JSON Directory: {JSON_DIR}")
    print(f"Categories CSV: {CATEGORIES_CSV}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print(f"Max requests per batch: {MAX_REQUESTS_PER_BATCH:,}")
    print(f"DCM Features: {', '.join(FEATURES.keys())}")

    # Create output directories
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    batch_files_dir = os.path.join(BASE_OUTPUT_DIR, "batch_files")
    os.makedirs(batch_files_dir, exist_ok=True)

    tracking_file = os.path.join(BASE_OUTPUT_DIR, "batch_tracking.json")

    # Load prompts and categories
    print("\nLoading DCM prompts...")
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} feature prompts: {', '.join(prompts.keys())}")

    print("\nLoading channel categories...")
    categories = load_categories(CATEGORIES_CSV)
    print(f"Found {len(categories)} channels in included categories")

    # Regenerate batch files with current prompts
    batch_tracking = process_all_data(JSON_DIR, categories, batch_files_dir, prompts)

    if not batch_tracking:
        print("\nNo batch files created. Exiting.")
        return

    # Save tracking before submission
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

    # Validate custom_id uniqueness across all batch files
    print("\nValidating custom_id uniqueness across batch files...")
    all_custom_ids = set()
    duplicates = 0
    for batch_info in batch_tracking:
        with open(batch_info['file_name'], 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                cid = obj['custom_id']
                if cid in all_custom_ids:
                    duplicates += 1
                    print(f"  DUPLICATE custom_id: {cid}")
                all_custom_ids.add(cid)
    if duplicates > 0:
        print(f"\nERROR: Found {duplicates} duplicate custom_ids! Aborting submission.")
        print("Fix the deduplication logic before resubmitting.")
        return
    print(f"  All {len(all_custom_ids):,} custom_ids are unique.")

    # Submit all batches
    batch_tracking = submit_batches(batch_tracking, tracking_file)

    # Final save
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline complete!")
    print(f"Track progress with: python {os.path.basename(__file__)} --status")
    print(f"Retrieve results with: python {os.path.basename(__file__)} --retrieve")


def check_status():
    """Check status of submitted batch jobs and automatically resubmit failed ones."""
    tracking_file = os.path.join(BASE_OUTPUT_DIR, "batch_tracking.json")

    if not os.path.exists(tracking_file):
        print(f"Error: Tracking file not found: {tracking_file}")
        return

    with open(tracking_file, 'r', encoding='utf-8') as f:
        batch_tracking = json.load(f)

    print(f"\n{'=' * 60}")
    print("Batch Job Status")
    print(f"{'=' * 60}\n")

    failed_batches = []
    not_submitted = []

    for batch_info in batch_tracking:
        batch_num = batch_info['batch_number']
        batch_job_id = batch_info.get('batch_job_id')

        if not batch_job_id:
            print(f"Batch {batch_num}: Not submitted")
            not_submitted.append(batch_info)
            continue

        try:
            batch = client.batches.retrieve(batch_job_id)
            status = batch.status
            completed = batch.request_counts.completed if batch.request_counts else 0
            failed = batch.request_counts.failed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else 0

            print(f"Batch {batch_num} ({batch_job_id}): {status}")
            print(f"  Progress: {completed}/{total} completed, {failed} failed")

            if batch.output_file_id:
                print(f"  Output file: {batch.output_file_id}")
                batch_info['output_file_id'] = batch.output_file_id

            batch_info['latest_status'] = status
            batch_info['checked_at'] = datetime.now().isoformat()

            if status == 'failed':
                failed_batches.append(batch_info)

            # Track completed batches that have failed tasks within them
            if (status == 'completed' and failed > 0
                    and batch.error_file_id
                    and not batch_info.get('failed_tasks_resubmitted')):
                batch_info['_error_file_id'] = batch.error_file_id
                batch_info['_failed_task_count'] = failed

        except Exception as e:
            print(f"Batch {batch_num}: Error checking status - {e}")

    # Save updated tracking
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

    # Combine failed and not-submitted batches for resubmission
    batches_to_resubmit = failed_batches + not_submitted
    if batches_to_resubmit:
        print(f"\n{'=' * 60}")
        print(f"Resubmitting {len(batches_to_resubmit)} batch(es) "
              f"({len(failed_batches)} failed, {len(not_submitted)} not submitted)")
        print(f"{'=' * 60}\n")

        resubmitted = 0
        for batch_info in batches_to_resubmit:
            batch_num = batch_info['batch_number']
            file_name = batch_info['file_name']

            if not os.path.exists(file_name):
                print(f"Batch {batch_num}: JSONL file not found ({file_name}), skipping")
                continue

            print(f"Resubmitting batch {batch_num}: {os.path.basename(file_name)} "
                  f"({batch_info['task_count']:,} tasks)")
            try:
                # Deduplicate the JSONL file before resubmitting
                seen_ids = set()
                deduped_lines = []
                with open(file_name, 'r', encoding='utf-8') as f:
                    for line in f:
                        obj = json.loads(line)
                        cid = obj['custom_id']
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            deduped_lines.append(line)
                original_count = batch_info['task_count']
                if len(deduped_lines) < original_count:
                    print(f"  Deduplicated: {original_count} -> {len(deduped_lines)} tasks "
                          f"({original_count - len(deduped_lines)} duplicates removed)")
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.writelines(deduped_lines)
                    batch_info['task_count'] = len(deduped_lines)

                with open(file_name, 'rb') as f:
                    batch_file = client.files.create(file=f, purpose='batch')

                batch_job = client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint='/v1/chat/completions',
                    completion_window="24h"
                )

                batch_info['file_id'] = batch_file.id
                batch_info['batch_job_id'] = batch_job.id
                batch_info['status'] = 'resubmitted'
                batch_info['latest_status'] = 'validating'
                batch_info['resubmitted_at'] = datetime.now().isoformat()
                batch_info.pop('error', None)
                batch_info.pop('output_file_id', None)
                batch_info.pop('downloaded', None)

                print(f"  -> Resubmitted: {batch_job.id}")
                resubmitted += 1

            except Exception as e:
                print(f"  -> Error resubmitting: {e}")
                batch_info['error'] = str(e)

            # Save after each resubmission
            with open(tracking_file, 'w', encoding='utf-8') as f:
                json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

        print(f"\nResubmitted: {resubmitted}/{len(batches_to_resubmit)}")
    else:
        print("\nNo failed or unsubmitted batches to resubmit.")

    # --- Resubmit individual failed TASKS from completed batches ---
    batches_with_failed_tasks = [
        b for b in batch_tracking if b.get('_error_file_id')
    ]

    if batches_with_failed_tasks:
        total_failed = sum(b['_failed_task_count'] for b in batches_with_failed_tasks)
        print(f"\n{'=' * 60}")
        print(f"Collecting {total_failed} failed task(s) from "
              f"{len(batches_with_failed_tasks)} completed batch(es)")
        print(f"{'=' * 60}\n")

        failed_task_lines = []

        for batch_info in batches_with_failed_tasks:
            batch_num = batch_info['batch_number']
            error_file_id = batch_info.pop('_error_file_id')
            failed_count = batch_info.pop('_failed_task_count')

            try:
                # Download error file to get failed custom_ids
                print(f"  Batch {batch_num}: {failed_count} failed task(s)")
                error_content = client.files.content(error_file_id)
                failed_custom_ids = set()
                for line in error_content.text.splitlines():
                    if line.strip():
                        error_result = json.loads(line)
                        cid = error_result.get('custom_id')
                        if cid:
                            failed_custom_ids.add(cid)

                # Find original request lines from the batch input file
                file_name = batch_info['file_name']
                if os.path.exists(file_name):
                    with open(file_name, 'r', encoding='utf-8') as f:
                        for line in f:
                            obj = json.loads(line)
                            if obj['custom_id'] in failed_custom_ids:
                                failed_task_lines.append(line)
                else:
                    print(f"    WARNING: Batch file not found: {file_name}")

                # Mark so we don't re-collect on next --status run
                batch_info['failed_tasks_resubmitted'] = True

            except Exception as e:
                print(f"    Error collecting failed tasks: {e}")

        if failed_task_lines:
            print(f"\nTotal failed tasks collected: {len(failed_task_lines)}")

            # Create new batch files respecting MAX_REQUESTS_PER_BATCH
            batch_files_dir = os.path.join(BASE_OUTPUT_DIR, "batch_files")
            max_batch_num = max(b['batch_number'] for b in batch_tracking)

            new_batches = []
            for i in range(0, len(failed_task_lines), MAX_REQUESTS_PER_BATCH):
                chunk = failed_task_lines[i:i + MAX_REQUESTS_PER_BATCH]
                new_num = max_batch_num + 1 + len(new_batches)
                file_name = os.path.join(
                    batch_files_dir, f"batch_{new_num:04d}_retry.jsonl")

                with open(file_name, 'w', encoding='utf-8') as f:
                    f.writelines(chunk)

                new_batch_info = {
                    'batch_number': new_num,
                    'file_name': file_name,
                    'task_count': len(chunk),
                    'is_retry': True,
                }
                new_batches.append(new_batch_info)
                print(f"  Created retry batch {new_num}: "
                      f"{len(chunk):,} tasks -> {os.path.basename(file_name)}")

            # Submit retry batches
            retry_submitted = 0
            for new_batch_info in new_batches:
                new_num = new_batch_info['batch_number']
                file_name = new_batch_info['file_name']

                try:
                    with open(file_name, 'rb') as f:
                        batch_file = client.files.create(file=f, purpose='batch')

                    batch_job = client.batches.create(
                        input_file_id=batch_file.id,
                        endpoint='/v1/chat/completions',
                        completion_window="24h"
                    )

                    new_batch_info['file_id'] = batch_file.id
                    new_batch_info['batch_job_id'] = batch_job.id
                    new_batch_info['status'] = 'submitted'
                    new_batch_info['submitted_at'] = datetime.now().isoformat()
                    print(f"    -> Submitted: {batch_job.id}")
                    retry_submitted += 1

                except Exception as e:
                    print(f"    -> Error: {e}")
                    new_batch_info['error'] = str(e)
                    new_batch_info['status'] = 'failed'

            # Add retry batches to tracking and save
            batch_tracking.extend(new_batches)
            with open(tracking_file, 'w', encoding='utf-8') as f:
                json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

            print(f"\nRetry batches submitted: {retry_submitted}/{len(new_batches)}")
        else:
            print("\nNo failed task lines found in batch files.")


def parse_custom_id(custom_id):
    """
    Parse the custom_id to extract components.
    Format: {category}_{channel}_{message_id}_{feature}
    Handles feature names containing underscores (e.g. outgroup_othering).
    """
    known_features = list(FEATURES.keys())
    feature = None
    prefix = custom_id
    for feat in sorted(known_features, key=len, reverse=True):
        if custom_id.endswith("_" + feat):
            feature = feat
            prefix = custom_id[:-(len(feat) + 1)]
            break

    if feature is None:
        return None

    # prefix is now "{category}_{channel}_{message_id}"
    parts = prefix.split('_')
    if len(parts) < 3:
        return None

    category = parts[0]
    message_id = parts[-1]
    channel = '_'.join(parts[1:-1])

    return {
        'category': category,
        'channel': channel,
        'message_id': message_id,
        'feature': feature
    }


def load_channel_messages(json_path):
    """Load all messages from a channel JSON file. Returns {message_id: msg_dict}."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error loading {json_path}: {e}")
        return {}

    messages = {}
    for msg in data.get('messages', []):
        if isinstance(msg, dict):
            msg_id = str(msg.get('message_id', ''))
            if msg_id:
                messages[msg_id] = msg
    return messages


def extract_message_metadata(msg):
    """Extract all relevant metadata from a message for the final CSV."""
    if not msg:
        return {}

    metadata = {}

    metadata['author'] = msg.get('author')
    metadata['message_type'] = msg.get('type')

    # Temporal data
    timestamp = msg.get('timestamp', {})
    if isinstance(timestamp, dict):
        metadata['timestamp_iso'] = timestamp.get('iso')
        metadata['date'] = timestamp.get('date')
        metadata['time'] = timestamp.get('time')
        metadata['timezone'] = timestamp.get('timezone')

    if metadata.get('date'):
        try:
            date_parts = metadata['date'].split('-')
            metadata['year'] = date_parts[0]
            metadata['month'] = date_parts[1]
        except Exception:
            pass

    # Content
    content = msg.get('content', {})
    if isinstance(content, dict):
        text = content.get('text') or content.get('text_plain', '')
        if isinstance(text, list):
            parts = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    t = part.get('text')
                    if isinstance(t, str):
                        parts.append(t)
            text = " ".join(parts)
        metadata['text'] = clean_text(text)

        # Media info
        media = content.get('media', [])
        if media:
            metadata['has_media'] = True
            metadata['media_count'] = len(media)
            metadata['media_types'] = ','.join(set(
                m.get('type', 'unknown') for m in media if isinstance(m, dict)))
        else:
            metadata['has_media'] = False
            metadata['media_count'] = 0
            metadata['media_types'] = ''

    # Conversation structure
    metadata['is_continuation'] = msg.get('is_continuation', False)
    metadata['reply_to_message_id'] = msg.get('reply_to_message_id')
    metadata['is_reply'] = msg.get('reply_to_message_id') is not None

    # Forwarding info
    forwarded = msg.get('forwarded', {})
    if isinstance(forwarded, dict):
        metadata['is_forwarded'] = forwarded.get('is_forwarded', False)
        metadata['forwarded_from'] = forwarded.get('original_sender')
        metadata['forwarded_timestamp'] = forwarded.get('original_timestamp')
    else:
        metadata['is_forwarded'] = False

    # Engagement - reactions
    engagement = msg.get('engagement', {})
    if isinstance(engagement, dict):
        reactions = engagement.get('reactions', [])
        metadata['total_reactions'] = 0
        reaction_counts = defaultdict(int)
        for reaction in reactions:
            if isinstance(reaction, dict):
                emoji = reaction.get('emoji', reaction.get('type', 'unknown'))
                count = reaction.get('count', 1)
                reaction_counts[emoji] += count
                metadata['total_reactions'] += count
            elif isinstance(reaction, str):
                reaction_counts[reaction] += 1
                metadata['total_reactions'] += 1
        for emoji, count in reaction_counts.items():
            metadata[f'reaction_{emoji}'] = count
    else:
        metadata['total_reactions'] = 0

    return metadata


def parse_gpt_response(response_str):
    """Parse GPT response to extract binary classification. Returns (raw, binary)."""
    import pandas as pd
    if not response_str or pd.isna(response_str):
        return None, None

    response_str = str(response_str).strip()

    if response_str in ['0', '1']:
        return response_str, int(response_str)

    response_lower = response_str.lower()
    if response_lower in ['yes', 'true']:
        return response_str, 1
    elif response_lower in ['no', 'false']:
        return response_str, 0

    if response_lower.startswith('1') or 'yes' in response_lower:
        return response_str, 1
    elif response_lower.startswith('0') or 'no' in response_lower:
        return response_str, 0

    return response_str, None


def retrieve():
    """Download results from completed batches (download only, no parsing)."""
    tracking_file = os.path.join(BASE_OUTPUT_DIR, "batch_tracking.json")
    results_dir = os.path.join(BASE_OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(tracking_file):
        print(f"Error: Tracking file not found: {tracking_file}")
        return

    with open(tracking_file, 'r', encoding='utf-8') as f:
        batch_tracking = json.load(f)

    print(f"\n{'=' * 60}")
    print("Downloading Batch Results")
    print(f"{'=' * 60}\n")

    total = len([b for b in batch_tracking if b.get('batch_job_id')])
    downloaded = 0
    already_done = 0
    not_ready = 0

    for batch_info in batch_tracking:
        batch_num = batch_info['batch_number']
        batch_job_id = batch_info.get('batch_job_id')
        if not batch_job_id:
            continue

        output_path = os.path.join(results_dir, f"results_batch_{batch_num:04d}.jsonl")

        if os.path.exists(output_path) and batch_info.get('downloaded'):
            already_done += 1
            downloaded += 1
            continue

        try:
            batch = client.batches.retrieve(batch_job_id)
            if batch.status != 'completed':
                print(f"Batch {batch_num}/{total}: {batch.status}")
                not_ready += 1
                continue
            if not batch.output_file_id:
                print(f"Batch {batch_num}/{total}: No output file")
                not_ready += 1
                continue

            print(f"Downloading batch {batch_num}/{total}...")
            content = client.files.content(batch.output_file_id)
            with open(output_path, 'wb') as f:
                f.write(content.read())

            batch_info['downloaded'] = True
            batch_info['output_path'] = output_path
            batch_info['downloaded_at'] = datetime.now().isoformat()
            downloaded += 1

        except Exception as e:
            print(f"Batch {batch_num}/{total}: Error - {e}")

        # Save tracking after each download (resume-safe)
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

    # Final save
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(batch_tracking, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("Download Summary")
    print(f"{'=' * 60}")
    print(f"  Already downloaded: {already_done}")
    print(f"  Newly downloaded:   {downloaded - already_done}")
    print(f"  Not ready:          {not_ready}")
    print(f"  Total downloaded:   {downloaded}/{total}")

    if not_ready > 0:
        print(f"\n{not_ready} batch(es) not yet completed. Run --retrieve again later.")
    if downloaded == total:
        print(f"\nAll batches downloaded. Run --parse to create the analysis CSV.")


def parse_results():
    """Parse downloaded batch results, merge metadata, create analysis CSV."""
    import pandas as pd
    import csv

    tracking_file = os.path.join(BASE_OUTPUT_DIR, "batch_tracking.json")
    results_dir = os.path.join(BASE_OUTPUT_DIR, "results")
    output_dir = os.path.join(BASE_OUTPUT_DIR, "parsed_results")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(tracking_file):
        print(f"Error: Tracking file not found: {tracking_file}")
        return

    with open(tracking_file, 'r', encoding='utf-8') as f:
        batch_tracking = json.load(f)

    # --- Step 1: Parse results from downloaded JSONL files ---
    print(f"\n{'=' * 60}")
    print("Parsing Results")
    print(f"{'=' * 60}\n")

    all_results = []
    parse_errors = 0

    for batch_info in batch_tracking:
        output_path = batch_info.get('output_path')
        if not output_path:
            output_path = os.path.join(
                results_dir, f"results_batch_{batch_info['batch_number']:04d}.jsonl")
        if not os.path.exists(output_path):
            continue

        file_count = 0
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    parsed = parse_custom_id(result.get('custom_id', ''))
                    if not parsed:
                        parse_errors += 1
                        continue

                    # Skip failed tasks within the batch
                    error = result.get('error')
                    if error:
                        parse_errors += 1
                        continue

                    response = result.get('response', {})
                    status_code = response.get('status_code')
                    if status_code and status_code != 200:
                        parse_errors += 1
                        continue

                    body = response.get('body', {})
                    choices = body.get('choices', [])
                    label = choices[0]['message']['content'].strip() if choices else None

                    if label is None:
                        parse_errors += 1
                        continue

                    all_results.append({
                        'category': parsed['category'],
                        'channel': parsed['channel'],
                        'message_id': parsed['message_id'],
                        'feature': parsed['feature'],
                        'label': label,
                    })
                    file_count += 1
                except json.JSONDecodeError:
                    parse_errors += 1

        print(f"  Batch {batch_info['batch_number']}: {file_count:,} results")

    print(f"\nTotal parsed: {len(all_results):,}")
    if parse_errors:
        print(f"Parse errors: {parse_errors}")

    if not all_results:
        print("No results to process.")
        return

    # --- Step 2: Pivot to wide format and merge metadata ---
    print(f"\n{'=' * 60}")
    print("Creating Analysis CSV")
    print(f"{'=' * 60}\n")

    df = pd.DataFrame(all_results)
    feature_list = list(FEATURES.keys())

    df_wide = df.pivot_table(
        index=['category', 'channel', 'message_id'],
        columns='feature',
        values='label',
        aggfunc='first'
    ).reset_index()

    for feature in feature_list:
        if feature not in df_wide.columns:
            df_wide[feature] = None

    print(f"Unique messages: {len(df_wide):,}")

    # Build stem -> channel_name lookup
    channel_names = {}
    with open(CATEGORIES_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ch = row.get('channel', '').strip()
            cat = row.get('category', '').strip()
            if ch and cat and cat in INCLUDED_CATEGORIES:
                stem = channel_to_filename(ch)
                if stem and stem not in channel_names:
                    channel_names[stem] = ch

    # Load original messages for metadata
    print("Loading message metadata...")
    channels = df_wide['channel'].unique()
    message_cache = {}
    for channel in channels:
        json_path = os.path.join(JSON_DIR, f"{channel}.json")
        if os.path.exists(json_path):
            message_cache[channel] = load_channel_messages(json_path)
        else:
            message_cache[channel] = {}

    print(f"Loaded {len(message_cache)} channel files")

    # Extract metadata per message
    print("Extracting metadata...")
    metadata_rows = []
    for _, row in df_wide.iterrows():
        channel = row['channel']
        message_id = str(row['message_id'])
        msg = message_cache.get(channel, {}).get(message_id, {})
        metadata = extract_message_metadata(msg)
        metadata['channel_name'] = channel_names.get(channel, channel)
        metadata_rows.append(metadata)

    metadata_df = pd.DataFrame(metadata_rows)
    df_final = pd.concat([df_wide.reset_index(drop=True), metadata_df], axis=1)

    # Parse GPT responses to binary
    print("Parsing GPT responses to binary...")
    for feature in feature_list:
        if feature in df_final.columns:
            df_final[f'{feature}_raw'] = df_final[feature]
            parsed = df_final[feature].apply(parse_gpt_response)
            df_final[f'{feature}_bin'] = parsed.apply(lambda x: x[1] if x else None)

    # Reorder columns
    id_cols = ['category', 'channel', 'channel_name', 'message_id']
    temporal_cols = ['date', 'year', 'month', 'time', 'timestamp_iso', 'timezone']
    content_cols = ['text', 'author']
    structure_cols = ['is_continuation', 'is_reply', 'reply_to_message_id']
    forward_cols = ['is_forwarded', 'forwarded_from', 'forwarded_timestamp']
    media_cols = ['has_media', 'media_count', 'media_types']
    engagement_cols = ['total_reactions']
    reaction_cols = [c for c in df_final.columns if c.startswith('reaction_')]

    dcm_cols = []
    for feature in feature_list:
        dcm_cols.extend([feature, f'{feature}_raw', f'{feature}_bin'])

    all_cols = (id_cols + temporal_cols + content_cols + structure_cols +
                forward_cols + media_cols + engagement_cols + reaction_cols + dcm_cols)
    final_cols = [c for c in all_cols if c in df_final.columns]
    remaining = [c for c in df_final.columns if c not in final_cols]
    final_cols.extend(remaining)
    df_final = df_final[final_cols]

    sort_cols = ['category', 'channel']
    if 'date' in df_final.columns:
        sort_cols.append('date')
    df_final = df_final.sort_values(sort_cols)

    output_file = os.path.join(output_dir, "telegram_dcm_annotations.csv")
    df_final.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"  Messages: {len(df_final):,}")
    print(f"  Columns: {len(df_final.columns)}")

    # --- Step 3: Coverage check against full source dataset ---
    print(f"\n{'=' * 60}")
    print("Coverage Check (annotations vs source data)")
    print(f"{'=' * 60}\n")

    # Build set of annotated (category, channel, message_id)
    annotated = set(
        zip(df_final['category'], df_final['channel'], df_final['message_id'].astype(str))
    )

    # Count all eligible messages from source JSON files
    categories_map = load_categories(CATEGORIES_CSV)
    processed_stems = set()
    total_source = 0
    missing_by_category = defaultdict(int)
    missing_by_channel = defaultdict(int)

    for channel_name, category in sorted(categories_map.items()):
        if channel_name.startswith('._'):
            continue
        stem = channel_to_filename(channel_name)
        if not stem:
            continue
        stem_key = (category, stem)
        if stem_key in processed_stems:
            continue
        processed_stems.add(stem_key)

        json_path = os.path.join(JSON_DIR, f"{stem}.json")
        if not os.path.exists(json_path):
            continue

        seen_ids = set()
        for message_id, text in iter_messages(json_path):
            if message_id in seen_ids:
                continue
            seen_ids.add(message_id)
            cleaned = clean_text(text)
            if len(cleaned.split()) < 3:
                continue
            total_source += 1
            if (category, stem, message_id) not in annotated:
                missing_by_category[category] += 1
                missing_by_channel[(category, stem)] += 1

    total_missing = sum(missing_by_category.values())
    print(f"  Source messages (eligible):  {total_source:,}")
    print(f"  Annotated messages:          {len(annotated):,}")

    if total_missing == 0:
        print(f"  ALL messages have annotations.")
    else:
        print(f"  MISSING annotations:         {total_missing:,}")
        print(f"\n  Missing by category:")
        for cat in sorted(missing_by_category):
            print(f"    {cat}: {missing_by_category[cat]:,}")
        # Show top channels with most missing
        top_missing = sorted(missing_by_channel.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top channels with missing annotations:")
        for (cat, ch), n in top_missing:
            print(f"    {cat}/{ch}: {n:,}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--submit-only':
            submit_only()
        elif sys.argv[1] == '--status':
            check_status()
        elif sys.argv[1] == '--retrieve':
            retrieve()
        elif sys.argv[1] == '--parse':
            parse_results()
    else:
        main()
