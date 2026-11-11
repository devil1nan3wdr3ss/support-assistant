"""Run eval/cases.csv against the real model and write a per-case report.

Usage (from project root, with SUPPORT_API_KEY / SUPPORT_MODEL / SUPPORT_API_URL set):
    python3 eval/run_eval.py [--only 01,05,27]

Auto-grading is a first pass only: every answered case still needs a human read
for fabrication, because keyword checks cannot judge meaning.
"""
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402

_orig_urlopen = urllib.request.urlopen
last_raw = {}


class DailyQuotaExhausted(Exception):
    pass


def recording_urlopen(req, timeout=None):
    """Record raw model output (even if validation later rejects it); retry 429/503."""
    for attempt in range(5):
        try:
            with _orig_urlopen(req, timeout=timeout) as r:
                body = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and b'PerDay' in e.read():
                raise DailyQuotaExhausted('当日免费额度用完')
            if e.code not in (429, 503) or attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    data = json.loads(body)
    last_raw['finish_reason'] = data['choices'][0].get('finish_reason')
    last_raw['content'] = data['choices'][0]['message'].get('content')
    return io.BytesIO(body)


urllib.request.urlopen = recording_urlopen


def norm(s):
    return s.replace(' ', '').lower()


def terms(cell):
    """'a;b|c' -> [['a'], ['b', 'c']]: every group must hit, any alternative in a group counts."""
    return [g.split('|') for g in cell.split(';') if g]


def grade(case, res, err):
    exp = case['expected_status']
    if err:
        return '调用/校验失败', err
    st, text = res['status'], norm(res['answer'])
    short = not last_raw  # no HTTP call recorded => answer() short-circuited before the model
    if exp == 'clarify':
        return ('澄清正确', '') if st == 'clarify' else ('路由错误', '期望 clarify，实际 ' + st)
    if exp == 'insufficient':
        if st == 'insufficient':
            return '拒答正确', '检索短路' if short else '模型判断'
        bad = [t for g in terms(case['must_not_contain']) for t in g if norm(t) in text]
        return ('疑似编造' if bad else '应拒未拒(待人工)'), '期望 insufficient，实际 ' + st + (' 含禁词 ' + str(bad) if bad else '')
    # expected answered
    if st == 'clarify':
        return '路由错误', '期望回答，实际 clarify'
    if st == 'insufficient':
        return '漏答', '检索短路（关键词未命中）' if short else '模型判为资料不足'
    want = set(filter(None, case['expected_citations'].split(';')))
    got = set(res['citations'])
    bad = [t for g in terms(case['must_not_contain']) for t in g if norm(t) in text]
    if bad:
        return '疑似编造', '含禁词 ' + str(bad)
    if want - got or got - want:
        return '引用错误', '缺 %s 多 %s' % (sorted(want - got), sorted(got - want))
    miss = [g for g in terms(case['must_contain']) if not any(norm(t) in text for t in g)]
    if miss:
        return '要点缺失(待人工)', '缺 ' + str(miss)
    return '答对', ''


def main():
    only = None
    if '--only' in sys.argv:
        only = set(sys.argv[sys.argv.index('--only') + 1].split(','))
    cases = list(csv.DictReader(open(ROOT / 'eval/cases.csv', encoding='utf-8-sig')))
    if only:
        cases = [c for c in cases if c['id'] in only]
    rows = []
    for c in cases:
        last_raw.clear()
        res, err = None, ''
        try:
            res = app.answer(c['question'], c['model'], c['language'], 'live')
        except DailyQuotaExhausted:
            print('%s 当日额度用完，停止；已跑 %d 题，用 --only 补跑剩下的' % (c['id'], len(rows)))
            break
        except Exception as e:  # keep going; a rejected answer is a result too
            err = '%s: %s' % (type(e).__name__, e)
        verdict, detail = grade(c, res, err)
        row = {'id': c['id'], 'category': c['category'], 'expected': c['expected_status'],
               'actual': res['status'] if res else 'error', 'citations': ';'.join(res['citations']) if res else '',
               'verdict': verdict, 'detail': detail, 'answer': res['answer'] if res else '',
               'raw_finish': last_raw.get('finish_reason', ''), 'raw_content': last_raw.get('content', '') if err else '',
               'question': c['question']}
        rows.append(row)
        print('%s %-6s %-14s %s' % (c['id'], row['actual'], verdict, detail), flush=True)
        time.sleep(1)

    if not rows:
        return
    stamp = datetime.now().strftime('%Y%m%d-%H%M')
    out = ROOT / 'eval/results'
    out.mkdir(exist_ok=True)
    path = out / ('run-%s.csv' % stamp)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ['human_verdict', 'human_note'])
        w.writeheader()
        w.writerows(rows)
    print('\n模型:', app.os.environ.get('SUPPORT_MODEL'), '| 题数:', len(rows))
    for k, v in Counter(r['verdict'] for r in rows).most_common():
        print('  %-16s %d' % (k, v))
    print('明细:', path)


if __name__ == '__main__':
    main()
