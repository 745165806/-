import tempfile
import unittest
from pathlib import Path
import numpy as np
import joblib
from ontoforensics.utterance.data import ROLES, validate
from ontoforensics.utterance.fusion import fit, evaluate, table
from scipy.special import expit
from ontoforensics.utterance.data import read, write, audio

class UtteranceTests(unittest.TestCase):
    def fixture(self):
        rows=[]; obs=[]
        rng=np.random.default_rng(123)
        for role in ['expert_train']+ROLES+['test']:
            for i in range(20):
                rid=f'{role}-{i}'
                rows.append(dict(recording_id=rid,role=role,label=i%2,dataset='fixture',group_id=rid,path=rid,row_index=''))
                for j,e in enumerate(['a','b','c']):
                    obs.append(dict(recording_id=rid,expert=e,score=float((i%2)*1.5+rng.normal()),
                                    quality=[-20.,0.],checkpoint_sha256=e,config_sha256='x',higher_is_fake=True,scope='utterance',status='valid'))
        ontology=dict(experts=['a','b','c'],families=dict(a='x',b='x',c='y'),relations=True,quality=True)
        return rows,obs,ontology
    def test_leakage(self):
        rows,_,_=self.fixture(); rows[-1]['group_id']=rows[0]['group_id']
        with self.assertRaisesRegex(ValueError,'leakage'): validate(rows)
    def test_evidence_conflict(self):
        rows,obs,o=self.fixture(); obs.append(dict(obs[0],score=123))
        with self.assertRaisesRegex(ValueError,'Conflicting'): table(rows,obs,o['experts'])
    def test_checkpoint_alias_is_not_an_independent_expert(self):
        rows,obs,o=self.fixture()
        for row in obs:
            if row['expert']=='b': row['checkpoint_sha256']='a'; row['config_sha256']='renamed-config'
        with self.assertRaisesRegex(ValueError,'Same model'):
            table(rows,obs,o['experts'])
    def test_contributions_and_isolation(self):
        rows,obs,o=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.joblib'
            fit(rows,obs,o,'relation_lr',model)
            first=joblib.load(model)
            # Changing target labels cannot affect model fitting.
            for r in rows:
                if r['role']=='test': r['label']=1-r['label']
            fit(rows,obs,o,'relation_lr',model)
            second=joblib.load(model)
            np.testing.assert_array_equal(first['classifier'].coef_,second['classifier'].coef_)
            evaluate(rows,obs,model,Path(directory)/'eval')
            from ontoforensics.utterance.data import read
            for r in read(Path(directory)/'eval/predictions.jsonl'):
                self.assertAlmostEqual(sum(r['contributions'].values())+r['intercept'],r['precalibration_logit'])
    def test_missing_expert_abstains(self):
        rows,obs,o=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.joblib'; fit(rows,obs,o,'flat_lr',model)
            obs=[r for r in obs if not (r['recording_id']=='test-0' and r['expert']=='a')]
            evaluate(rows,obs,model,Path(directory)/'eval')
            from ontoforensics.utterance.data import read
            r=read(Path(directory)/'eval/predictions.jsonl')[0]
            self.assertIsNone(r['probability'])

    def test_entire_expert_missing_and_empty_evidence_abstain(self):
        rows,obs,o=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.joblib'; fit(rows,obs,o,'relation_lr',model)
            for evidence in ([r for r in obs if r['expert']!='a'],[]):
                metrics=evaluate(rows,evidence,model,Path(directory)/'eval')
                predictions=read(Path(directory)/'eval/predictions.jsonl')
                self.assertTrue(all(r['probability'] is None for r in predictions))
                self.assertEqual(metrics['fixture']['coverage'],0)

    def test_external_alias_cannot_reuse_fitted_group_location_or_content(self):
        import copy
        rows,obs,o=self.fixture()
        rows[0]['audio_sha256']='known-content'
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.joblib'; fit(rows,obs,o,'flat_lr',model)
            for key,value in [('group_id',rows[0]['group_id']),('path',rows[0]['path']),('audio_sha256','known-content')]:
                changed=copy.deepcopy(rows)
                changed[-1][key]=value
                with self.assertRaisesRegex(ValueError,'leakage'):
                    evaluate(changed,obs,model,Path(directory)/'eval')

    def test_test_evidence_is_not_inspected_during_fitting(self):
        rows,obs,o=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.joblib'; fit(rows,obs,o,'relation_lr',model)
            first=joblib.load(model)
            for r in obs:
                if r['recording_id'].startswith('test-'):
                    r['score']=float('nan'); r['checkpoint_sha256']='different-test-model'
            fit(rows,obs,o,'relation_lr',model)
            second=joblib.load(model)
            np.testing.assert_array_equal(first['classifier'].coef_,second['classifier'].coef_)
            self.assertEqual(first['sources'],second['sources'])

    def test_calibrated_contributions_reconstruct_probability_and_unlabelled_prediction(self):
        rows,obs,o=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.joblib'; fit(rows,obs,o,'relation_lr',model)
            for row in rows:
                if row['role']=='test': row['label']=None; row['role']='unspecified'
            metrics=evaluate(rows,obs,model,Path(directory)/'eval',predict=True)
            self.assertEqual(metrics['fixture']['n'],0)
            for row in read(Path(directory)/'eval/predictions.jsonl'):
                logit=sum(row['calibrated_contributions'].values())+row['calibrated_intercept']
                self.assertAlmostEqual(float(expit(logit)),row['probability'])
                self.assertEqual(len(row['evidence']),3)

    def test_all_variants_run_with_one_expert(self):
        rows,obs,o=self.fixture(); o['experts']=['a']
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('mean','flat_lr','mlp','relation_lr','pairwise_lr'):
                model=Path(directory)/(mode+'.joblib'); fit(rows,obs,o,mode,model)
                metrics=evaluate(rows,obs,model,Path(directory)/mode)
                self.assertEqual(metrics['fixture']['n'],20)

    def test_score_version_change_rejected_on_evaluation(self):
        rows,obs,o=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.joblib'; fit(rows,obs,o,'flat_lr',model)
            for row in obs: row['checkpoint_sha256']+='changed'
            with self.assertRaisesRegex(ValueError,'version changed'):
                evaluate(rows,obs,model,Path(directory)/'eval')

    def test_graph_does_not_publish_labels(self):
        try:
            from rdflib import Graph
        except ImportError:
            self.skipTest('rdflib optional')
        from ontoforensics.utterance.graph import export_graph
        rows,obs,o=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.joblib'; fit(rows,obs,o,'relation_lr',model)
            evaluate(rows,obs,model,Path(directory)/'eval')
            graphfile=Path(directory)/'evidence.ttl'
            export_graph(read(Path(directory)/'eval/predictions.jsonl'),graphfile)
            graph=Graph().parse(graphfile,format='turtle')
            self.assertGreater(len(graph),100)
            self.assertFalse(any('label' in str(p).lower() for _,p,_ in graph))

class WaveformTests(unittest.TestCase):
    def test_interrupted_training_resumes_to_identical_weights(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch optional')
        import soundfile as sf
        from unittest.mock import patch
        from ontoforensics.utterance.experts import train
        rows,_,_=UtteranceTests().fixture()
        rows=[r for r in rows if int(r['recording_id'].rsplit('-',1)[1])<2]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i,r in enumerate(rows):
                path=root/(r['recording_id']+'.wav')
                sf.write(path,np.sin(np.arange(2048)*(.05+.001*i))*.1,16000)
                r.update(path=str(path),storage='file')
            config=dict(expert='lfcc_cnn',samples=1024,epochs=2,patience=3,micro_batch=2,effective_batch=2,cpu_threads=1)
            train(rows,config,root/'continuous')
            with patch('ontoforensics.utterance.experts.write',side_effect=RuntimeError('simulated interruption')):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    train(rows,config,root/'resumed')
            train(rows,config,root/'resumed',resume=root/'resumed/last.pt')
            a=torch.load(root/'continuous/last.pt',weights_only=False)['state_dict']
            b=torch.load(root/'resumed/last.pt',weights_only=False)['state_dict']
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))

    def test_embedded_audio_selects_correct_row_for_each_storage(self):
        import io
        import soundfile as sf
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('pyarrow optional')
        payloads=[]
        for value in (.1,.2,.3):
            buffer=io.BytesIO(); sf.write(buffer,np.full(1024,value),8000,format='WAV',subtype='FLOAT')
            payloads.append({'bytes':buffer.getvalue(),'path':None})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); table=pa.table({'audio':payloads})
            pq.write_table(table,root/'audio.parquet',row_group_size=1)
            for kind in ('file','stream'):
                with pa.OSFile(str(root/(kind+'.arrow')),'wb') as sink:
                    writer=ipc.new_file(sink,table.schema) if kind=='file' else ipc.new_stream(sink,table.schema)
                    with writer:
                        for batch in table.to_batches(max_chunksize=1): writer.write_batch(batch)
            for file,storage in [('audio.parquet','parquet_embedded'),('file.arrow','arrow_embedded'),('stream.arrow','arrow_embedded')]:
                x=audio(dict(path=str(root/file),storage=storage,row_index='2'))
                self.assertEqual(len(x),2048)
                self.assertAlmostEqual(float(x[100:-100].mean()),.3,places=3)
                with self.assertRaises(ValueError): audio(dict(path=str(root/file),storage=storage,row_index='-1'))

    def test_real_waveform_training_export_and_resume(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch optional')
        import soundfile as sf
        from ontoforensics.utterance.experts import train, export, build, score
        rows,_,_=UtteranceTests().fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            # Audio content varies by recording; roles do not reuse paths.
            for i,r in enumerate(rows):
                path=root/(r['recording_id']+'.wav')
                x=.2*np.sin(np.arange(2048)*(.1+.001*i))
                sf.write(path,x,16000); r.update(path=str(path),storage='file')
            # Keep a small but valid full-role protocol.
            rows=[r for r in rows if int(r['recording_id'].rsplit('-',1)[1])<2]
            config=dict(expert='lfcc_cnn',samples=1024,epochs=1,micro_batch=2,effective_batch=2,cpu_threads=1)
            train(rows,config,root/'model')
            report=export(rows,root/'model/best.pt',root/'scores.jsonl')
            self.assertEqual(report['count'],14)
            self.assertTrue(all(np.isfinite(r['score']) for r in read(root/'scores.jsonl')))
            self.assertEqual(len(train(rows,config,root/'model',resume=root/'model/last.pt')),1)
            changed=dict(config,lr=.9)
            with self.assertRaisesRegex(ValueError,'signature changed'):
                train(rows,changed,root/'model',resume=root/'model/last.pt')

    def test_duration_weighted_chunk_aggregation(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch optional')
        import soundfile as sf
        from ontoforensics.utterance.experts import score
        class MeanExpert(torch.nn.Module):
            def forward(self,x): return torch.stack([x.mean(1)*0,x.mean(1)],1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audio.wav'
            sf.write(path,np.r_[np.zeros(1024),np.ones(256)*.5],16000,subtype='FLOAT')
            value,_=score(MeanExpert(),dict(path=str(path),storage='file'),1024,'cpu')
            self.assertAlmostEqual(value,.1,places=6)

if __name__=='__main__': unittest.main()
