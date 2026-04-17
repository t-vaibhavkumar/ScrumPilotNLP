import sys
sys.path.insert(0, '.')

print('=== ScrumPilot End-to-End Import Check ===\n')

checks = [
    ('Unit 1 - Tokenizer',    'from backend.nlp.unit1_preprocessing.tokenizer import sentence_tok'),
    ('Unit 1 - NER',          'from backend.nlp.unit1_preprocessing.ner import extract_assignees'),
    ('Unit 1 - TF-IDF',       'from backend.nlp.unit1_representations.tfidf import build_tfidf'),
    ('Unit 2 - LSTM',         'from backend.nlp.unit2_models.lstm_classifier import LSTMClassifier, predict as lstm_predict'),
    ('Unit 2 - GRU',          'from backend.nlp.unit2_models.gru_classifier import GRUClassifier, predict as gru_predict'),
    ('Unit 3 - SBERT',        'from backend.nlp.unit3_transformers.sentence_bert import find_matching_story'),
    ('Unit 4 - Summarizer',   'from backend.nlp.unit4_applications.summarizer import extractive_summarize'),
    ('Context Loader',        'from backend.nlp.context_loader import load_sprint_context, SprintContext'),
    ('Jira Mapper',           'from backend.nlp.jira_action_mapper import map_standup_approval_payload, map_sprint_planning'),
    ('Orchestrator',          'from backend.nlp.pipeline_orchestrator import NLPOrchestrator, get_orchestrator'),
    ('DB CRUD (context)',     'from backend.db.crud import get_active_sprint, get_sprint_stories_with_details, get_sprint_assignments'),
    ('Jira Client',           'from backend.tools.jira_client import JiraManager'),
]

ok = 0
for name, stmt in checks:
    try:
        exec(stmt)
        print('  OK   ' + name)
        ok += 1
    except Exception as e:
        print('  FAIL ' + name + ': ' + str(e)[:80])

print('\nResult: ' + str(ok) + '/' + str(len(checks)) + ' components ready')

if ok >= 11:  # only need first 11 (Jira client optional)
    import torch, pickle, os
    from backend.nlp.unit2_models.lstm_classifier import LSTMClassifier, predict as lp
    from backend.nlp.unit2_models.gru_classifier import GRUClassifier, predict as gp
    M = 'backend/nlp/models'
    lv = pickle.load(open(M+'/lstm_meeting_type.vocab','rb'))
    gv = pickle.load(open(M+'/gru_action_type.vocab','rb'))
    lm = LSTMClassifier(vocab_size=len(lv), embed_dim=128, num_classes=3, num_layers=1, dropout=0.3)
    lm.load_state_dict(torch.load(M+'/lstm_meeting_type.pt', weights_only=True))
    gm = GRUClassifier(vocab_size=len(gv), num_classes=5, dropout=0.5)
    gm.load_state_dict(torch.load(M+'/gru_action_type.pt', weights_only=True))
    r1 = lp(lm, lv, 'I am blocked waiting for Docker credentials from DevOps no blockers')
    r2 = gp(gm, gv, 'Alice will take the authentication story this sprint')
    print('\nFunctional inference test:')
    print('  LSTM: ' + r1['prediction'] + '  conf=' + str(round(r1['confidence'],2)))
    print('  GRU:  ' + r2['prediction'] + '  conf=' + str(round(r2['confidence'],2)))
    print('\nScrumPilot is end-to-end ready.')
