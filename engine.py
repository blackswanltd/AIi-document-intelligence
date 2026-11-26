"""Synthetic document classification and lexical retrieval demo."""
import json, re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parent
TOPICS = {
 'invoice': ['invoice payment balance', 'billing amount payable', 'purchase invoice due', 'vendor payment total'],
 'insurance': ['insurance policy claim', 'coverage premium deductible', 'claim adjuster benefits', 'policy renewal coverage'],
 'compliance': ['compliance audit controls', 'regulatory inspection findings', 'risk assessment policy', 'audit corrective action'],
 'logistics': ['shipment delivery tracking', 'warehouse inventory dispatch', 'freight carrier transport', 'delivery route shipment']}

def dataset():
    return [{'id': f'{label}-{i}', 'label': label, 'text': f'{phrases[i%4]} reference REF-{i:03d} amount ${100+i}.00 date 2026-10-{i%28+1:02d}'} for label,phrases in TOPICS.items() for i in range(40)]

def extract(text):
    return {'references': re.findall(r'\bREF-\d+\b',text), 'amounts': re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?',text), 'dates': re.findall(r'\b\d{4}-\d{2}-\d{2}\b',text)}

def classifier(rows):
    model=make_pipeline(TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'), LogisticRegression(max_iter=1000,random_state=42))
    return model.fit([r['text'] for r in rows],[r['label'] for r in rows])

def search(rows, query, limit=3):
    if not query.strip() or not rows or limit<=0: return []
    vectorizer=TfidfVectorizer(); matrix=vectorizer.fit_transform([r['text'] for r in rows]); scores=cosine_similarity(vectorizer.transform([query]),matrix)[0]
    return [{'id':rows[i]['id'],'text':rows[i]['text'],'score':float(scores[i])} for i in sorted(range(len(rows)),key=lambda i:-scores[i])[:limit] if scores[i]>0]

def run():
    rows=dataset(); train,test=train_test_split(rows,test_size=.25,random_state=42,stratify=[r['label'] for r in rows]); model=classifier(train); y=[r['label'] for r in test]; pred=model.predict([r['text'] for r in test]); result={'documents':len(rows),'train':len(train),'test':len(test),'accuracy':accuracy_score(y,pred),'report':classification_report(y,pred,output_dict=True,zero_division=0),'example_extraction':extract(rows[0]['text']),'example_search':search(rows,'warehouse inventory')}; (ROOT/'documents.json').write_text(json.dumps(rows,indent=2)); (ROOT/'results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
