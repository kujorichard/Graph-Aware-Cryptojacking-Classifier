import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock
import geval_batch as g

SOURCE=Path(__file__).parent/'example_analysis.json'

class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.inputs=self.root/'input'; self.inputs.mkdir()
        self.outputs=self.root/'output'; self.repeats=self.root/'repeats'
        self.original=json.loads(SOURCE.read_text())
        self.cfg=g.Config(api_key='synthetic-test-key',retry_delay=0,max_attempts=2)
    def tearDown(self): self.tmp.cleanup()
    def write(self,name='one.json',binary='sample.exe',data=None,directory=None):
        value=copy.deepcopy(data if data is not None else self.original)
        value['sample']['binary_name']=binary
        path=(directory or self.inputs)/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8'); return path
    def form(self,item,score=3):
        return {'sample_id':item['sample_id'],'scores':dict.fromkeys(g.DIMENSIONS,score),
            'justifications':dict.fromkeys(g.DIMENSIONS,'SYNTHETIC TEST ONLY. Not actual evaluation.'),
            'unsupported_claims':[], 'shap_faithfulness':{'faithful':True,'reason':'Synthetic fixture only.','discrepancies':[]},
            'evidence_limitations':['Synthetic fixture.'], 'summary':'Synthetic test; no actual scores assigned.'}
    def response(self,body,cfg):
        text=body['messages'][0]['content'].split('UNTRUSTED SAMPLE:\n',1)[1].split('\nPrior output',1)[0]
        item=json.loads(text)
        return {'model':'claude-sonnet-5-5','stop_reason':'end_turn',
            'content':[{'type':'thinking','thinking':'synthetic summary','signature':'synthetic'},
                       {'type':'text','text':json.dumps(self.form(item))}],
            'usage':{'input_tokens':10,'output_tokens':20}}
    def runquiet(self,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return g.run_batch(self.inputs,self.outputs,self.repeats,self.cfg,expected_files=None,**kwargs)
    def test_261_file_batch_and_resume(self):
        for i in range(261): self.write(f'{i:03}.json',f'binary_{i}.exe')
        with patch.object(g,'send_request',side_effect=self.response) as api:
            result=self.runquiet()
        self.assertEqual(api.call_count,261)
        self.assertEqual(result['successfully_evaluated'],261)
        self.assertEqual(result['table_4_13'][0]['Count'],261)
        self.assertEqual(result['table_4_13'][0]['Percentage'],100)
        self.assertEqual(result['table_4_16'][0]['Mean score'],3)
        self.assertEqual(result['table_4_16'][0]['SD'],0)
        with patch.object(g,'send_request',side_effect=AssertionError('Resume must not call API')) as api:
            resumed=self.runquiet()
        self.assertEqual(api.call_count,0)
        self.assertEqual(resumed['successfully_evaluated'],261)
        self.assertEqual(len(list((self.outputs/'results').glob('*.json'))),261)
        for filename in ['report.html','report.md','summary.json','per_sample_scores.csv',
            'table_4_13_hallucination.csv','table_4_14_faithfulness.csv','table_4_15_consistency.csv','table_4_16_quality.csv',
            'unsupported_claims.csv','faithfulness_discrepancies.csv','consistency_details.json','batch_manifest.json']:
            self.assertTrue((self.outputs/filename).is_file())
    def test_dry_run_makes_no_requests(self):
        self.write(); self.cfg.api_key=''
        with patch.object(g,'send_request',side_effect=AssertionError('No dry-run calls')):
            result=self.runquiet(dry_run=True)
        self.assertEqual(result['counts'],{'input_ready':1})
        self.assertEqual(result['successfully_evaluated'],0)
        self.assertIsNone(result['table_4_13'][0]['Percentage'])
        self.assertIsNone(result['table_4_16'][0]['Mean score'])
    def test_api_request_matches_sonnet_55_protocol(self):
        item,_=g.normalize(self.original,self.cfg); body=g.request_body(item,self.cfg)
        self.assertEqual(body['model'],'claude-sonnet-5-5')
        self.assertEqual(body['thinking'],{'type':'adaptive','display':'summarized'})
        self.assertEqual(body['output_config']['format']['type'],'json_schema')
        for invalid in ['temperature','top_p','top_k','output_format']: self.assertNotIn(invalid,body)
        self.assertNotIn('budget_tokens',json.dumps(body))
        self.assertNotIn('minimum',json.dumps(body['output_config']['format']['schema']))
        self.cfg.thinking=False
        self.assertEqual(g.request_body(item,self.cfg)['thinking'],{'type':'between_tools'})
    def test_quote_schema_and_score_validation(self):
        item,_=g.normalize(self.original,self.cfg); form=self.form(item)
        form['unsupported_claims']=[{'quote':'No evidence points toward benign behavior.',
            'candidate_path':'/candidate_analysis/explanation','evidence_reference':'/evidence/full_attributions',
            'reason':'Only a top-k subset is supplied.','category':'unsupported_certainty'}]
        g.validate_evaluation(form,item)
        for score in [None,True,0,6,2.5,'3']:
            bad=copy.deepcopy(form); bad['scores']['technical_accuracy']=score
            with self.assertRaises(ValueError): g.validate_evaluation(bad,item)
        form['unsupported_claims'][0]['quote']='Never written in candidate'
        with self.assertRaises(ValueError): g.validate_evaluation(form,item)
    def test_original_values_pass_rounding_audit_and_blinding(self):
        item,private=g.normalize(self.original,self.cfg)
        audit=g.numeric_audit(item,self.cfg)
        self.assertTrue(audit['numeric_match']); self.assertEqual(audit['checked_assessments'],20)
        body=json.dumps(g.request_body(item,self.cfg))
        for hidden in ['actual_label','actual_class','dataset/cryptojacking','1d42789ea54be34db','generator_backend','generator_model']:
            self.assertNotIn(hidden,body)
        self.assertNotIn('generator',item)
        self.assertEqual(item['candidate_analysis'],self.original['llm']['analysis'])
        self.assertEqual(private['sample_metadata']['actual_label'],1)
    def test_numeric_discrepancy_overrides_judge_faithfulness(self):
        data=copy.deepcopy(self.original); data['llm']['analysis']['evidence_analysis'][0]['shap_contribution']=-.011675
        self.write(data=data)
        with patch.object(g,'send_request',side_effect=self.response): result=self.runquiet()
        self.assertEqual(result['table_4_14'][1]['Count'],1)
        self.assertEqual(result['table_4_13'][1]['Count'],1)
        record=g.read_json(next((self.outputs/'results').glob('*.json')))
        self.assertFalse(record['faithful_to_shap']); self.assertTrue(record['has_unsupported_claims'])
    def test_sample_sd(self):
        item,_=g.normalize(self.original,self.cfg)
        rows=[{'status':'success','has_unsupported_claims':False,'faithful_to_shap':True,'evaluation':self.form(item,score=x)} for x in [1,5]]
        cons=g.consistency({},self.repeats,self.cfg); summary=g.summarize(rows,cons,{})
        self.assertEqual(summary['table_4_16'][0]['Mean score'],3)
        self.assertAlmostEqual(summary['table_4_16'][0]['SD'],math.sqrt(8))
        rows=rows[:1]; self.assertIsNone(g.summarize(rows,cons,{})['table_4_16'][0]['SD'])
    def test_bad_inputs_generation_errors_and_duplicates(self):
        self.write('a.json'); self.write('b.json')
        (self.inputs/'bad.json').write_text('{bad')
        data=copy.deepcopy(self.original); data['llm']['status']='failed'; self.write('failed.json','failed.exe',data)
        with patch.object(g,'send_request',side_effect=self.response) as api: summary=self.runquiet()
        self.assertEqual(api.call_count,1)
        self.assertEqual(summary['counts'],{'success':1,'duplicate_input':1,'input_error':1,'generation_error':1})
    def test_invalid_final_retry(self):
        self.write(); sequence=[{'content':[{'type':'text','text':'{broken'}],'stop_reason':'end_turn'},None]
        def reply(body,cfg):
            first=sequence.pop(0)
            return first if first is not None else self.response(body,cfg)
        with patch.object(g,'send_request',side_effect=reply) as api: summary=self.runquiet()
        self.assertEqual(api.call_count,2); self.assertEqual(summary['successfully_evaluated'],1)
        self.assertIn('Prior output failed validation',api.call_args.args[0]['messages'][0]['content'])
    def test_thinking_only_and_truncation_never_scored(self):
        for stop in ['end_turn','max_tokens','refusal','model_context_window_exceeded']:
            with self.subTest(stop=stop),tempfile.TemporaryDirectory() as temp:
                item,private=g.normalize(self.original,self.cfg)
                with patch.object(g,'send_request',return_value={'content':[{'type':'thinking','thinking':json.dumps(self.form(item))}],'stop_reason':stop}):
                    result=g.evaluate(item,private,self.cfg,Path(temp)/'result.json','hash')
                self.assertEqual(result['status'],'judge_error'); self.assertIsNone(result['evaluation'])
    def test_rate_limit_retry(self):
        self.write(); calls=0
        def reply(body,cfg):
            nonlocal calls; calls+=1
            if calls==1: raise g.ProviderError('rate limit',429,0)
            return self.response(body,cfg)
        with patch.object(g,'send_request',side_effect=reply): summary=self.runquiet()
        self.assertEqual(calls,2); self.assertEqual(summary['successfully_evaluated'],1)
    def test_fatal_api_error_stops_and_reports_unattempted(self):
        for i in range(4): self.write(f'{i}.json',f'{i}.exe')
        with patch.object(g,'send_request',side_effect=g.ProviderError('unauthorized',401)) as api: summary=self.runquiet()
        self.assertEqual(api.call_count,1)
        self.assertEqual(summary['counts'],{'judge_error':1,'not_attempted_batch_stopped':3})
        self.assertIsNone(summary['table_4_13'][0]['Percentage'])
    def test_secret_redaction_and_no_key_in_config(self):
        self.write()
        with patch.object(g,'send_request',side_effect=g.ProviderError('bad '+self.cfg.api_key,401)): self.runquiet()
        for path in self.outputs.rglob('*'):
            if path.is_file(): self.assertNotIn(self.cfg.api_key,path.read_text(encoding='utf-8-sig'))
        self.assertNotIn(self.cfg.api_key,repr(self.cfg))
    def test_absent_repeat_data_not_100_percent(self):
        item,_=g.normalize(self.original,self.cfg); cons=g.consistency({item['sample_id']:item},self.repeats,self.cfg)
        self.assertEqual(cons['eligible_repeated_samples'],0)
        self.assertIsNone(cons['rows'][2]['Value']); self.assertIsNone(cons['rows'][3]['Value'])
    def test_repeat_tier_and_function_agreement(self):
        data=copy.deepcopy(self.original); data['llm']['analysis']['cited_functions']=['FUN_00401000','FUN_00402000']
        item,_=g.normalize(data,self.cfg)
        for run in [2,3]: self.write('repeat.json',data['sample']['binary_name'],data,self.repeats/f'run_{run}')
        cons=g.consistency({item['sample_id']:item},self.repeats,self.cfg)
        self.assertEqual(cons['rows'][2]['Value'],100); self.assertEqual(cons['rows'][3]['Value'],100)
        data['llm']['analysis']['tier']='low-confidence'; data['llm']['analysis']['cited_functions']=['FUN_00401000']
        self.write('repeat.json',data['sample']['binary_name'],data,self.repeats/'run_3')
        cons=g.consistency({item['sample_id']:item},self.repeats,self.cfg)
        self.assertEqual(cons['rows'][2]['Value'],0); self.assertEqual(cons['rows'][3]['Value'],0)
    def test_no_functions_and_changed_repeat_evidence(self):
        item,_=g.normalize(self.original,self.cfg)
        for run in [2,3]: self.write('repeat.json',self.original['sample']['binary_name'],directory=self.repeats/f'run_{run}')
        cons=g.consistency({item['sample_id']:item},self.repeats,self.cfg)
        self.assertEqual(cons['rows'][2]['Value'],100); self.assertIsNone(cons['rows'][3]['Value'])
        changed=copy.deepcopy(self.original); changed['shap']['top_features'][0]['shap_value']+=.1
        self.write('repeat.json',self.original['sample']['binary_name'],changed,self.repeats/'run_3')
        cons=g.consistency({item['sample_id']:item},self.repeats,self.cfg)
        self.assertEqual(cons['eligible_repeated_samples'],0); self.assertIsNone(cons['rows'][2]['Value'])
    def test_changed_protocol_requires_new_output(self):
        self.write()
        with patch.object(g,'send_request',side_effect=self.response): self.runquiet()
        self.cfg.effort='medium'
        with self.assertRaises(ValueError),patch.object(g,'send_request',side_effect=AssertionError('No new requests')): self.runquiet()
    def test_interruption_then_resume(self):
        self.write('a.json','a.exe'); self.write('b.json','b.exe')
        calls=0
        def reply(body,cfg):
            nonlocal calls; calls+=1
            if calls==2: raise KeyboardInterrupt()
            return self.response(body,cfg)
        with patch.object(g,'send_request',side_effect=reply): partial=self.runquiet()
        self.assertTrue(partial['metadata']['interrupted']); self.assertEqual(partial['successfully_evaluated'],1)
        with patch.object(g,'send_request',side_effect=self.response) as api: finished=self.runquiet()
        self.assertEqual(api.call_count,1); self.assertEqual(finished['successfully_evaluated'],2)
    def test_input_bom_and_duplicate_json_keys(self):
        path=self.inputs/'bom.json'; path.write_text(json.dumps(self.original),encoding='utf-8-sig')
        self.assertIsInstance(g.read_json(path),dict)
        path.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError): g.read_json(path)
    def test_report_escapes_html_and_spreadsheet_formulas(self):
        self.write('<script>.json','a.exe')
        with patch.object(g,'send_request',side_effect=self.response): self.runquiet()
        self.assertNotIn('<script>',(self.outputs/'report.html').read_text())
        g.csv_write(self.root/'safe.csv',['text'],[{'text':'=HYPERLINK("bad")'}])
        self.assertIn("'=HYPERLINK",(self.root/'safe.csv').read_text(encoding='utf-8-sig'))
    def test_http_endpoint_and_headers(self):
        item,_=g.normalize(self.original,self.cfg); body=g.request_body(item,self.cfg)
        response=MagicMock(); response.__enter__.return_value.read.return_value=b'{"content":[],"stop_reason":"end_turn"}'
        with patch.object(g.urllib.request,'urlopen',return_value=response) as opener:
            g.send_request(body,self.cfg)
        request=opener.call_args.args[0]
        self.assertEqual(request.full_url,'https://api.anthropic.com/v1/messages')
        self.assertEqual(request.get_header('Anthropic-version'),'2023-06-01')
        self.assertEqual(request.get_header('X-api-key'),self.cfg.api_key)
        self.assertEqual(json.loads(request.data),body)
    def test_corrupt_cache_recovered(self):
        self.write()
        with patch.object(g,'send_request',side_effect=self.response): self.runquiet()
        result_path=next((self.outputs/'results').glob('*.json')); result_path.write_text('{broken')
        with patch.object(g,'send_request',side_effect=self.response) as api: summary=self.runquiet()
        self.assertEqual(api.call_count,1); self.assertEqual(summary['successfully_evaluated'],1)
        self.assertEqual(len(list((self.outputs/'archived_results').rglob('*.json'))),1)
    def test_duplicate_repeat_group_excluded(self):
        item,_=g.normalize(self.original,self.cfg)
        self.write('a.json',self.original['sample']['binary_name'],directory=self.repeats/'run_2')
        self.write('duplicate.json',self.original['sample']['binary_name'],directory=self.repeats/'run_2')
        self.write('a.json',self.original['sample']['binary_name'],directory=self.repeats/'run_3')
        cons=g.consistency({item['sample_id']:item},self.repeats,self.cfg)
        self.assertEqual(cons['eligible_repeated_samples'],0)
        self.assertEqual(len(cons['errors']),1)
    def test_invalid_combined_components_rejected(self):
        for field in ['sample','random_forest','shap','llm']:
            data=copy.deepcopy(self.original); data[field]=[]
            with self.subTest(field=field),self.assertRaises(g.InputRejected): g.normalize(data,self.cfg)

if __name__=='__main__': unittest.main(verbosity=2)
