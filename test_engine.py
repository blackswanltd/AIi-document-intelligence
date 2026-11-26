import unittest
from engine import dataset, extract, classifier, search
class TestEngine(unittest.TestCase):
    def test_dataset(self): self.assertEqual(len(dataset()),160)
    def test_unique_ids(self): self.assertEqual(len({r['id'] for r in dataset()}),160)
    def test_extract(self): self.assertEqual(extract('REF-123 $25.00 2026-10-01'),{'references':['REF-123'],'amounts':['$25.00'],'dates':['2026-10-01']})
    def test_empty_extract(self): self.assertEqual(extract('hello')['amounts'],[])
    def test_classification(self): self.assertEqual(classifier(dataset()).predict(['warehouse inventory dispatch'])[0],'logistics')
    def test_search(self): self.assertIn('warehouse',search(dataset(),'warehouse inventory')[0]['text'])
    def test_empty_search(self): self.assertEqual(search(dataset(),''),[])
if __name__=='__main__': unittest.main()
