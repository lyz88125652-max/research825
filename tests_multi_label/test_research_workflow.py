"""Synthetic-only tests of search, leakage boundaries, frozen splits and final sealing."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from main_multi_label import parse_args, MODEL_REGISTRY
from multi_label_pipeline.data import frozen_splits
from multi_label_pipeline.ensemble import rank_subsets, average
from multi_label_pipeline.metrics import score
from multi_label_pipeline.stacking import LabelwiseStacker
from multi_label_pipeline.workflow import run, restore_args, cache_trainer

ROOT = Path(__file__).resolve().parents[1]


def synthetic(folder):
    patterns = np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,1,0],[1,1,1]])
    y = np.tile(patterns, (20,1))
    letters = "ACDEFGHIKLMNPQRSTVWY"
    sequences = np.array(["AC"+letters[i//20]+letters[i%20]+"A"*(i%7) for i in range(len(y))])
    rows = np.arange(2,len(y)+2)
    X = np.zeros((len(y),1280),dtype=np.float32)
    X[:,0] = np.arange(len(y)); X[:,1:4] = y
    source, features = folder/"labels.csv", folder/"features.npz"
    with source.open("w",newline="") as handle:
        writer=csv.writer(handle); writer.writerow(["sequence","antioxidant","antimicrobial","toxic"])
        writer.writerows((sequence,*target) for sequence,target in zip(sequences,y))
    np.savez(features,X=X,y=y,sequences=sequences,source_rows=rows,
             label_names=np.asarray(["antioxidant","antimicrobial","toxic"]),task=np.asarray("multilabel"))
    features.with_suffix(".metadata.json").write_text(json.dumps({
        "model":"facebook/esm2_t33_650M_UR50D","layer":33,"features":1280,"reused_records":0,
        "pooling":"mean over residue tokens; exclude BOS, EOS and padding"}))
    return source,features,X,y,sequences,rows


class ResearchTests(unittest.TestCase):
    def test_all_511_and_tie_breaks(self):
        y=np.array([[1,0,1],[0,1,0],[1,1,0],[0,0,1]])
        names=[f"model{i}" for i in range(9)]
        p=np.repeat((.1+.8*y)[:,None,:],9,axis=1)
        ranking=rank_subsets(y,p,names)
        self.assertEqual(len(ranking),511)
        self.assertEqual(sum(row["model_count"]==1 for row in ranking),9)
        self.assertEqual([row["model_count"] for row in ranking[:9]],[1]*9)
        np.testing.assert_allclose(average(p,names,[names[0],names[3]]),p[:,0])
        self.assertFalse(np.allclose(average(p,names,names).sum(1),1))

    def test_frozen_split_rejects_data_and_partition_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); _,_,_,y,sequences,rows=synthetic(folder)
            records=list(zip(rows.tolist(),sequences.tolist(),[tuple(target) for target in y.tolist()]))
            file=folder/"splits.json"
            first=frozen_splits(file,records,3,42,.1)
            with mock.patch("multi_label_pipeline.data.make_holdout",side_effect=AssertionError("Regenerated split")):
                second=frozen_splits(file,records,3,42,.1)
            np.testing.assert_array_equal(first[0],second[0])
            with self.assertRaises(ValueError): frozen_splits(file,records,3,43,.1)
            with self.assertRaises(ValueError): frozen_splits(file,records[::-1],3,42,.1)
            saved=json.loads(file.read_text()); saved["validation_indices"][0].append(saved["validation_indices"][1][0])
            file.write_text(json.dumps(saved))
            with self.assertRaises(ValueError): frozen_splits(file,records,3,42,.1)

    def test_stacker_three_label_features_and_brier(self):
        y=np.tile([[1,0,1],[0,1,0],[1,1,0],[0,0,1]],(8,1))
        p=np.repeat((.1+.8*y)[:,None,:],2,axis=1)
        stacker=LabelwiseStacker().fit(p,y,["a","b"])
        self.assertEqual([est.n_features_in_ for est in stacker.estimators_],[2,2,2])
        prediction=stacker.predict_proba(p,["a","b"])
        self.assertEqual(prediction.shape,y.shape)
        with self.assertRaises(ValueError): stacker.predict_proba(p,["b","a"])
        overall,classes=score(y,.1+.8*y)
        self.assertAlmostEqual(overall["brier_macro"],.01)
        self.assertAlmostEqual(classes[0]["brier"],.01)

    def test_stages_keep_test_locked_and_resume_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); source,features,X,y,sequences,rows=synthetic(folder)
            calls=[]
            def trainer(Xfit,yfit,Xvalid,groups,output,args,seed):
                calls.append((output,list(groups),Xvalid[:,0].astype(int).tolist()))
                self.assertEqual(len(set(groups)),len(groups))
                if Xfit.shape[1]==1:
                    p=np.full((len(Xvalid),3),.5)
                else: p=.1+.8*Xvalid[:,1:4]
                return p,{"synthetic":True}
            args=parse_args(["--stage","all","--models","lr","rf","svm","--input",str(source),
                "--features",str(features),"--split-file",str(folder/"splits.json"),"--folds","3",
                "--inner-folds","2","--top-k","3","--output-dir",str(folder/"results")])
            trainers={name:trainer for name in ["lr","rf","svm","length_lr"]}
            output=run(args,ROOT,MODEL_REGISTRY,trainers)
            manifest=json.loads((output/"splits.json").read_text())
            test=set(manifest["test_indices"]); train=set(manifest["training_indices"])
            for path,groups,predicted in calls:
                if "/bias/" not in str(path): self.assertFalse(test & set(predicted))
                self.assertFalse({str(sequences[i]) for i in test} & set(groups))
            for provenance in (output/"nested").glob("outer_*/inner/fold_provenance.json"):
                outer=json.loads((provenance.parents[1]/"partition.json").read_text())
                fit=set(outer["outer_fit_rows"]); valid=set(outer["outer_validation_rows"])
                for part in json.loads(provenance.read_text()):
                    self.assertTrue(set(part["fit_source_rows"])<=fit)
                    self.assertTrue(set(part["validation_source_rows"])<=fit)
                    self.assertFalse(valid & set(part["fit_source_rows"]))
                    self.assertFalse(valid & set(part["validation_source_rows"]))
            self.assertFalse((output/"final_access.json").exists())
            self.assertTrue((output/"final_recipe.json").exists())
            self.assertFalse(list(output.glob("*/test_predictions.csv")))
            before=len(calls)
            resume=parse_args(["--stage","cv","--run-dir",str(output)])
            run(resume,ROOT,MODEL_REGISTRY,trainers)
            self.assertEqual(len(calls),before,"CV did not reuse completed fits")
            # Recreate a selection seal after recomputing the same CV artifacts.
            run(parse_args(["--stage","select","--run-dir",str(output)]),ROOT,MODEL_REGISTRY,trainers)
            run(parse_args(["--stage","final","--run-dir",str(output)]),ROOT,MODEL_REGISTRY,trainers)
            final_calls=calls[before:]
            self.assertEqual(len(final_calls),3)
            for _,groups,predicted in final_calls:
                self.assertEqual(set(predicted),test)
                self.assertEqual(set(groups),{str(sequences[i]) for i in train})
            self.assertEqual(json.loads((output/"final_access.json").read_text())["status"],"complete")
            self.assertTrue((output/"final/test_roc.png").exists())
            with self.assertRaises(FileExistsError):
                run(parse_args(["--stage","final","--run-dir",str(output)]),ROOT,MODEL_REGISTRY,trainers)
            with self.assertRaises(ValueError):
                run(parse_args(["--stage","ensemble","--run-dir",str(output)]),ROOT,MODEL_REGISTRY,trainers)

    def test_reused_feature_guard_and_resume_options(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); source,features,*_=synthetic(folder)
            metadata=json.loads(features.with_suffix(".metadata.json").read_text()); metadata["reused_records"]=120
            features.with_suffix(".metadata.json").write_text(json.dumps(metadata))
            args=parse_args(["--models","lr","--input",str(source),"--features",str(features),
                            "--split-file",str(folder/"splits.json"),"--output-dir",str(folder/"results"),"--folds","3"])
            with self.assertRaisesRegex(ValueError,"fresh ESM-2"): run(args,ROOT,MODEL_REGISTRY,{})
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--run-dir","anything","--models","lr"])

    def test_extractor_backup_preserves_previous_outputs(self):
        spec=importlib.util.spec_from_file_location("shared_extractor",ROOT/"ESM-2.py")
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); source=folder/"labels.csv"; output=folder/"features.npz"
            source.write_text("sequence,antioxidant,antimicrobial,toxic\nAAA,1,0,0\nCCC,0,1,1\n")
            output.write_bytes(b"previous features")
            output.with_suffix(".metadata.json").write_text("previous metadata")
            output.with_suffix(".skipped.csv").write_text("previous skip report")
            with mock.patch.object(module,"extract",return_value=(np.ones((2,1280),dtype=np.float32),"test")):
                self.assertEqual(module.main(["--input",str(source),"--output",str(output)]),0)
            backup=next((folder/"backups").iterdir())
            self.assertEqual((backup/"features.npz").read_bytes(),b"previous features")
            self.assertEqual((backup/"features.metadata.json").read_text(),"previous metadata")
            self.assertEqual((backup/"features.skipped.csv").read_text(),"previous skip report")
            metadata=json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["reused_records"],0)
            self.assertEqual(metadata["extraction_kind"],"fresh_forward")


if __name__=="__main__": unittest.main()
