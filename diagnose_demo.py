import sys, os
sys.path.insert(0, r'e:\HalluDet\Hallu_Detect')
import numpy as np, torch
from src.phase1_nli.predict import NLIPredictor
from src.phase2_uncertainty.ensemble import EnsembleDisagreement
from src.phase3_rav.retriever import LightweightRAV
from src.phase4_ensemble.meta_learner import MetaLearnerPredictor
from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity
import spacy
from src.phase1_nli.claim_splitter import split_claims

print('Loading models...')
nli  = NLIPredictor()
ens  = EnsembleDisagreement(nli_predictor=nli, load_small=True, load_bart=True)
rav  = LightweightRAV(nli_predictor=nli)
meta = MetaLearnerPredictor()
ppl  = TinyLLaMaPerplexity()
nlp_sm = spacy.load('en_core_web_sm')

CASES = [
    {'label':'FACTUAL','knowledge':'Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect, not for his theory of relativity.','response':'Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.','expected':0},
    {'label':'HALLUCINATED','knowledge':'Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect, not for his theory of relativity.','response':'Albert Einstein won the Nobel Prize in Physics in 1922 for his theory of relativity.','expected':1},
    {'label':'FACTUAL','knowledge':'DNA stands for Deoxyribonucleic Acid. It does not stand for Dynamic Nucleic Assembly.','response':'DNA stands for Deoxyribonucleic Acid.','expected':0},
    {'label':'HALLUCINATED','knowledge':'DNA stands for Deoxyribonucleic Acid. It does not stand for Dynamic Nucleic Assembly.','response':'DNA stands for Dynamic Nucleic Assembly.','expected':1},
]

correct_total = 0
for case in CASES:
    for claim in split_claims(case['response']):
        rav_r = rav.verify_claim(claim)
        nr = nli.score_claim(case['knowledge'], claim)
        er = ens.score(case['knowledge'], claim)
        pr = ppl.score_claim(claim)
        doc = nlp_sm(claim)
        feat = np.array([nr['probs']['entailment'],nr['probs']['contradiction'],er['mean_entailment'],er['disagreement'],min(pr['log_perplexity'],10.0),rav_r['max_ent'],max((e['retrieval_score'] for e in rav_r.get('evidence_used',[])),default=0.0),float(np.log1p(len(doc.ents)))],dtype=np.float32)
        sc = meta.scaler.transform(feat.reshape(1,-1))
        with torch.no_grad(): raw = meta.model(torch.tensor(sc).float()).item()
        cal = float(meta.platt.predict_proba([[raw]])[0,1])
        ok = (cal>=0.5)==bool(case['expected'])
        if ok: correct_total += 1
        print('CASE=%s  NLI_ent=%.4f contra=%.4f  MLP=%.4f  PLATT=%.4f  PRED=%s  OK=%s' % (
            case['label'], nr['probs']['entailment'], nr['probs']['contradiction'],
            raw, cal, 'HALL' if cal>=0.5 else 'FACT', ok))

print('\nSummary: %d/4 correct (%.0f%%)' % (correct_total, 100*correct_total/4))
