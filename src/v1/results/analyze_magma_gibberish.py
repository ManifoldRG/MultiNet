import json
import os
import re
from typing import Any, Dict, Iterable, List, Set, Tuple
import csv


ROOT = os.path.join('src', 'v1', 'results', 'magma')


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


INSTRUCTION_VERBS = {
    'put', 'place', 'move', 'open', 'close', 'pick', 'grasp', 'stack', 'clean', 'press',
    'go', 'grab', 'turn', 'push', 'pull', 'drop', 'enter', 'exit', 'walk', 'run', 'done',
}


def looks_like_instruction(s: str) -> bool:
    s_low = s.lower()
    if '\n' in s or ';' in s or '->' in s:
        return True
    if re.search(r'\b(then|next|first|second|third|finally)\b', s_low):
        return True
    if re.search(r'^\s*\d+\s*[-.)]', s_low):  # numbered steps
        return True
    if any(v in s_low for v in INSTRUCTION_VERBS) and len(s_low.split()) >= 4:
        return True
    return False


def _is_numeric_token(tok: str) -> bool:
    # Accept integers or floats with optional sign
    return re.fullmatch(r'[+-]?\d+(?:\.\d+)?', tok) is not None


def is_numeric_only(s: Any) -> bool:
    """Return True if the string consists only of number tokens
    (e.g., '0', '-1', '12', '0.5', or sequences like '1 2 3')."""
    if not isinstance(s, str):
        return False
    s_norm = normalize_text(s)
    if not s_norm:
        return False
    # Split on whitespace and commas
    tokens = re.split(r'[\s,]+', s_norm)
    if not tokens:
        return False
    return all(_is_numeric_token(t) for t in tokens)


def is_coordinate_like(s: Any) -> bool:
    """Detect common coordinate-like outputs in text.
    Matches forms like:
      - 'Coordinate: (0.50, 0.15)'
      - '[119,37]'
      - 'Mark 1 at [136,37]'
      - '(12, 34)'
    """
    if not isinstance(s, str):
        return False
    s_norm = normalize_text(s)
    if not s_norm:
        return False
    low = s_norm.lower()
    # Keywords indicating coordinates
    if 'coordinate' in low or 'mark' in low:
        return True
    # Parentheses or brackets with two numeric components
    if re.search(r"\(\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*\)", s_norm):
        return True
    if re.search(r"\[\s*\d+\s*,\s*\d+\s*\]", s_norm):
        return True
    # Multiple bracketed coordinate-like pairs
    if re.search(r"\[(?:\s*\d+\s*,\s*\d+\s*\]){1,}", s_norm):
        return True
    return False


def is_gibberish_default(s: Any) -> bool:
    # Non-string or empty
    if not isinstance(s, str):
        return True
    s_norm = normalize_text(s)
    if not s_norm:
        return True

    # Overly long free-form text (likely sentences/instructions)
    if len(s_norm) > 60 or len(s_norm.split()) > 8:
        return True

    # Contains unusual punctuation patterns or repeated symbols
    if re.search(r'[{}\[\]<>]|_{2,}|\*{2,}|={2,}|/{2,}', s_norm):
        return True

    # Looks like multi-step instruction or contains imperative verbs heavily
    if looks_like_instruction(s_norm):
        return True

    # Pure digits or alphanumeric codes can be suspicious for VQA-like answers
    # We flag pure numbers and mixed digit-heavy tokens (e.g., "A12B")
    if re.fullmatch(r'\d+', s_norm):
        return True
    if re.search(r'\d', s_norm) and len(s_norm) <= 3:
        return True

    # Excess punctuation not typical of labels
    if re.search(r'[,:;.!?]', s_norm):
        return True

    return False


def collect_outputs(obj: Any) -> List[str]:
    outs: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Only consider 'all_outs' per request; ignore 'preds'
            if k == 'all_outs' and isinstance(v, list):
                outs.extend([x for x in v if isinstance(x, (str, int, float))])
            else:
                outs.extend(collect_outputs(v))
    elif isinstance(obj, list):
        for item in obj:
            outs.extend(collect_outputs(item))
    return outs


def _extract_key_recursive(obj: Any, key: str) -> List[Any]:
    vals: List[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                vals.append(v)
            vals.extend(_extract_key_recursive(v, key))
    elif isinstance(obj, list):
        for it in obj:
            vals.extend(_extract_key_recursive(it, key))
    return vals


def derive_expected_format(fp: str, data: Any) -> str:
    path_low = fp.lower()
    labels_lists = _extract_key_recursive(data, 'gt_labels')
    labels: List[Any] = []
    for lst in labels_lists:
        if isinstance(lst, list):
            labels.extend(lst)
    # Heuristics based on dataset and label value types
    if not labels:
        # Fallback by dataset name
        if 'robovqa' in path_low:
            return 'concise action phrase/instruction'
        if 'sqa3d' in path_low:
            return 'short normalized token (object/attribute/yes-no/count)'
        if 'piqa' in path_low:
            return 'binary choice (A/B) or option text'
        if 'odinw' in path_low:
            return 'class index (0..N-1) or textual class label'
        return 'short categorical text or index'

    types = {type(x) for x in labels}
    if types == {str}:
        # Distinguish short labels vs instructions by average word count
        avg_words = sum(len(str(x).split()) for x in labels) / max(1, len(labels))
        if 'robovqa' in path_low or avg_words > 4:
            return 'concise action phrase/instruction'
        if 'sqa3d' in path_low:
            return 'short normalized token (object/attribute/yes-no/count)'
        return 'short categorical text label'
    if types <= {int}:
        # Integer indices
        unique_vals = set(int(x) for x in labels if isinstance(x, int))
        if 'piqa' in path_low and unique_vals.issubset({0, 1}):
            return 'binary choice index (0/1); prefer A/B text'
        if 'odinw' in path_low:
            return 'class index (0..N-1) or textual class label'
        return 'categorical index (integer)'
    # Mixed or other
    return 'mixed labels; short text or indices'


def analyze_file(fp: str) -> Tuple[Set[str], int, str]:
    with open(fp, 'r') as f:
        data = json.load(f)
    outs = collect_outputs(data)
    # Dataset-aware behavior
    path_low = fp.lower()
    is_vqa = any(key in path_low for key in ['robovqa', 'sqa3d', 'piqa', 'odinw'])
    is_odinw = 'odinw' in path_low
    if is_vqa:
        # For VQA datasets:
        # - ODINW: allow single integer outputs; only flag coordinate-like strings
        # - Others (RoboVQA, PIQA, SQA3D): flag numeric-only and coordinate-like
        if is_odinw:
            gib = {normalize_text(o) for o in outs if is_coordinate_like(o)}
        else:
            gib = {normalize_text(o) for o in outs if is_numeric_only(o) or is_coordinate_like(o)}
    else:
        # For non-VQA datasets, use default broader gibberish heuristic
        gib = {normalize_text(o if isinstance(o, str) else str(o)) for o in outs if is_gibberish_default(o)}
    exp_fmt = derive_expected_format(fp, data)
    return gib, len(outs), exp_fmt


def main():
    json_files: List[str] = []
    for dirpath, _, filenames in os.walk(ROOT):
        for fn in filenames:
            if fn.endswith('.json'):
                json_files.append(os.path.join(dirpath, fn))

    print(f'Found {len(json_files)} magma result JSON files under {ROOT}')

    # Aggregate per dataset
    per_dataset: Dict[str, Dict[str, int]] = {}
    per_dataset_format: Dict[str, Set[str]] = {}
    per_file_summary: List[Tuple[str, int, int]] = []  # (file, num_gib, total_outs)

    def dataset_name_from_path(p: str) -> str:
        pl = p.replace('\\', '/').lower()
        # ODINW sub-dataset
        if '/odinw/' in pl and p.endswith('.json'):
            base = os.path.splitext(os.path.basename(p))[0]
            return f'odinw/{base}'
        # Common single-file datasets
        mapping = {
            'robovqa_results.json': 'robovqa',
            'sqa3d_results.json': 'sqa3d',
            'piqa_results.json': 'piqa',
            'magma_overcooked_results.json': 'overcooked',
        }
        for fname, name in mapping.items():
            if p.endswith(fname):
                return name
        # OpenX and others
        if 'openx' in pl:
            return 'openx'
        return os.path.splitext(os.path.basename(p))[0]

    # Walk files and aggregate
    total_gib_unique = 0
    for fp in sorted(json_files):
        try:
            gib_set, total, exp_fmt = analyze_file(fp)
        except Exception as e:
            print(f'- {fp}: ERROR parsing ({e})')
            continue
        per_file_summary.append((fp, len(gib_set), total))
        ds = dataset_name_from_path(fp)
        if ds not in per_dataset:
            per_dataset[ds] = {}
            per_dataset_format[ds] = set()
        for g in gib_set:
            per_dataset[ds][g] = per_dataset[ds].get(g, 0) + 1
        if exp_fmt:
            per_dataset_format[ds].add(exp_fmt)
        total_gib_unique += len(gib_set)

    # Emit stdout summary
    for fp, n_gib, total in per_file_summary:
        print(f'- {fp}: {n_gib} unique gibberish candidates (out of {total} outputs)')
    print(f'Total unique gibberish candidates across files: {total_gib_unique}')

    # Write JSON report per dataset
    report_json_path = os.path.join('src', 'v1', 'results', 'magma_gibberish_report.json')
    report = []
    for ds in sorted(per_dataset.keys()):
        entries = per_dataset[ds]
        # Sort tokens for stable output
        toks = sorted(entries.items(), key=lambda kv: (kv[1] * -1, kv[0]))
        fmts = sorted(per_dataset_format.get(ds, set()))
        report.append({
            'dataset': ds,
            'expected_format': ' | '.join(fmts) if fmts else 'unknown',
            'total_unique_gibberish': len(entries),
            'tokens': [{'text': t, 'count_files': c} for t, c in toks]
        })
    with open(report_json_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Write CSV report (dataset, token, count_files)
    report_csv_path = os.path.join('src', 'v1', 'results', 'magma_gibberish_report.csv')
    with open(report_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'token', 'count_files'])
        for ds in sorted(per_dataset.keys()):
            for tok, cnt in sorted(per_dataset[ds].items(), key=lambda kv: (kv[1] * -1, kv[0])):
                w.writerow([ds, tok, cnt])

    # Write expected formats CSV
    formats_csv_path = os.path.join('src', 'v1', 'results', 'magma_expected_formats.csv')
    with open(formats_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'expected_format'])
        for ds in sorted(per_dataset_format.keys()):
            w.writerow([ds, ' | '.join(sorted(per_dataset_format[ds]))])

    print(f'Wrote per-dataset JSON report to {report_json_path}')
    print(f'Wrote per-dataset CSV report to  {report_csv_path}')
    print(f'Wrote expected formats CSV to      {formats_csv_path}')


if __name__ == '__main__':
    main()
