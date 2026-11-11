import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer, model_answer


class SupportTests(unittest.TestCase):
    def test_missing_model(self):
        self.assertEqual(answer('设备离线')['status'], 'clarify')

    def test_model_in_question(self):
        self.assertEqual(answer('NX-200 power')['model'], 'NX-200')

    def test_conflict(self):
        self.assertEqual(answer('NX-200 power', 'NX-100')['status'], 'clarify')

    def test_isolation(self):
        for model, expected in [('NX-100', '12 V'), ('NX-200', '24 V')]:
            result = answer('供电电压', model)
            self.assertIn(expected, result['answer'])
            self.assertTrue(all(s['model'] == model for s in result['sources']))

    def test_unknown(self):
        result = answer('保修多久', 'NX-100')
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['sources'], [])

    def test_english(self):
        self.assertIn('15 seconds', answer('factory reset', 'NX-200', 'English')['answer'])

    def test_live_never_silently_falls_back(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                answer('power', 'NX-100', 'English', 'live')

    def test_truncated_output_rejected(self):
        # Real Gemini run: reasoning tokens hit max_tokens, JSON cut mid-string.
        body = io.BytesIO(json.dumps({'choices': [{'finish_reason': 'length', 'message': {
            'content': '{"status": "insufficient", "answer": "根据现有资料'}}]}).encode())
        env = {'SUPPORT_API_KEY': 'test', 'SUPPORT_MODEL': 'test'}
        with patch.dict('os.environ', env), patch('urllib.request.urlopen', return_value=body):
            with self.assertRaisesRegex(ValueError, 'length'):
                answer('power', 'NX-100', '中文', 'live')

    def _live(self, content):
        body = io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(content, ensure_ascii=False)}}]}).encode())
        env = {'SUPPORT_API_KEY': 'test', 'SUPPORT_MODEL': 'test'}
        with patch.dict('os.environ', env), patch('urllib.request.urlopen', return_value=body):
            return answer('NX-100 power', 'NX-100', '中文', 'live')

    def test_empty_answer_rejected(self):
        # Eval cases 23/41: {"status": "insufficient", "answer": ""} used to pass.
        with self.assertRaisesRegex(ValueError, '格式'):
            self._live({'status': 'insufficient', 'answer': '  ', 'citations': []})

    def test_insufficient_with_citations_rejected(self):
        # Eval cases 29/31/42: insufficient + citations contradicts the protocol.
        with self.assertRaisesRegex(ValueError, '矛盾'):
            self._live({'status': 'insufficient', 'answer': '不支持 PoE', 'citations': ['N100-POWER']})

    def test_valid_live_answer_passes(self):
        r = self._live({'status': 'answered', 'answer': '12 V DC、1 A', 'citations': ['N100-POWER']})
        self.assertEqual([s['id'] for s in r['sources']], ['N100-POWER'])

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            answer('power', 'OTHER')


if __name__ == '__main__':
    unittest.main()
