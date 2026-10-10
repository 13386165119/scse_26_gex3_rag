"""Control-flow tests with synthetic generation. NOT real model evaluation."""
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from agent import PolicyAgent, UNKNOWN_PHONE, UNKNOWN_GENERAL, enforce_grounding
from rag_tool import PolicyRetriever

PHONE = 'What exact phone number should I call for the University IT Service Desk?'
DOCS = [Document(page_content='Reset your password in the university account portal. Contact the IT Service Desk if your account remains locked.', metadata={'policy_id': 'test-1', 'topic': 'accounts'})]


class AgentTests(unittest.TestCase):
    def test_model_reviewer_and_revision_are_called(self):
        agent = PolicyAgent()
        with patch.object(agent, '_generate', side_effect=['Call 5551234567.', 'REVISE: The number is absent.', 'The phone number is not specified.']) as generation:
            self.assertEqual(agent.answer(PHONE, DOCS), UNKNOWN_PHONE)
            self.assertEqual(generation.call_count, 3)
            self.assertTrue(agent.last_trace['model_called'])
            self.assertEqual(agent.last_trace['draft'], 'Call 5551234567.')
            self.assertTrue(agent.last_trace['safeguard_applied'])

    def test_known_policy_is_not_replaced_with_phone_refusal(self):
        agent = PolicyAgent()
        with patch.object(agent, '_generate', side_effect=['Use the university account portal.', 'PASS']):
            self.assertEqual(agent.answer('How do I reset my password?', DOCS), 'Use the university account portal.')
            self.assertFalse(agent.last_trace['safeguard_applied'])

    def test_no_evidence_does_not_claim_inference(self):
        agent = PolicyAgent()
        with patch.object(agent, '_generate') as generate:
            self.assertEqual(agent.answer(PHONE, []), UNKNOWN_PHONE)
            generate.assert_not_called()
            self.assertFalse(agent.last_trace['model_called'])

    def test_unrelated_number_and_irrelevant_advice_rejected(self):
        for evidence in ['Library phone: 5551234567', 'Service Desk case number: 5551234567', DOCS[0].page_content]:
            for proposed in ['Call 5551234567.', 'Check the website.']:
                self.assertEqual(enforce_grounding(PHONE, evidence, proposed), UNKNOWN_PHONE)

    def test_bm25_validation(self):
        retriever = PolicyRetriever(DOCS)
        self.assertTrue(retriever.search('reset password'))
        self.assertEqual(retriever.search('galactic starship'), [])
        with self.assertRaises(ValueError):
            retriever.search('reset password', top_k=0)


if __name__ == '__main__':
    unittest.main()
