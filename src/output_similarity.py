import logging
from typing import Any

import bert_score
import nltk
import numpy as np
import sacrebleu
import torch
from cer import calculate_cer
from nltk import word_tokenize
from nltk.tokenize.nist import NISTTokenizer
from nltk.translate import meteor_score
from nltk.translate.nist_score import corpus_nist
from rouge_score import rouge_scorer, scoring
from sentence_transformers import SentenceTransformer, SimilarityFunction
from tqdm import tqdm

from src.utils import get_gpu_utilization

logger = logging.getLogger(__name__)

logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

nltk.download('perluniprops', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)


def preprocess_similarity_inputs(
    predictions: list[list[str]], 
    references: list[list[str]],
) -> tuple[list[str], list[list[str]]]:
    """
    Prepares predictions and references for pairwise metric calculation.

    This function flattens the list of prediction groups and aligns the references
    accordingly. For example, if predictions are `[[p1_s1, p1_s2], [p2_s1]]` and
    references are `[[r1], [r2]]`, it returns `([p1_s1, p1_s2, p2_s1], [[r1], [r1], [r2]])`.

    Args:
        predictions (list[list[str]]): A list of prediction groups, where each inner list
                                       contains one or more generated sequences for a single prompt.
        references (list[list[str]]): A list of reference groups, where each inner list
                                      contains one or more ground-truth strings.

    Returns:
        (tuple[list[str], list[list[str]]]): A tuple containing the flattened list of predictions
                                             and the corresponding aligned list of references.
    """
    if not predictions:
        return [], []
    
    # If predictions are [[pred1_seq1, pred1_seq2], [pred2_seq1]],
    # and references are [[ref1], [ref2]],
    # flat_predictions = [pred1_seq1, pred1_seq2, pred2_seq1]
    # cloned_references = [[ref1], [ref1], [ref2]]
    
    flat_predictions = []
    cloned_references_aligned = []

    if len(predictions) != len(references):
        raise ValueError(f"Number of prediction groups ({len(predictions)}) must match number of reference groups ({len(references)}).")

    for i, pred_list_for_sample in enumerate(predictions):
        if not pred_list_for_sample:
            logger.warning(f"Sample {i} has no predictions. Skipping.")
            continue
        
        ref_for_sample = references[i]
        if not ref_for_sample:
            logger.warning(f"Sample {i} has no references. Skipping corresponding predictions.")
            continue

        for pred_str in pred_list_for_sample:
            flat_predictions.append(pred_str)
            cloned_references_aligned.append(ref_for_sample)
            
    return flat_predictions, cloned_references_aligned


def sacrebleu_score(
    predictions: list[list[str]],
    references: list[list[str]]
) -> tuple[float, int]:
    """Computes the corpus-level SacreBLEU score."""
    flat_predictions, cloned_references = preprocess_similarity_inputs(predictions, references)
    if not flat_predictions: return np.nan, get_gpu_utilization()
    transformed_references = list(zip(*cloned_references))
    transformed_references = [list(r) for r in transformed_references]
    score = np.nan
    try:
        result = sacrebleu.corpus_bleu(
            flat_predictions,
            transformed_references,
            smooth_method="exp",
            smooth_value=None,
            force=False,
            lowercase=False,
            tokenize=None,
            use_effective_order=False,
        )
        score = result.score
    except Exception as e:
        logger.warning(f"Error in sacrebleu: {e}")
    return score, get_gpu_utilization()


def rouge(
    predictions: list[list[str]],
    references: list[list[str]]
) -> tuple[dict[str, float], int]:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores."""
    flat_predictions, cloned_references = preprocess_similarity_inputs(predictions, references)
    if not flat_predictions: return {rt: np.nan for rt in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}, get_gpu_utilization()

    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types)
    aggregator = scoring.BootstrapAggregator()
    
    for pred, ref_list in zip(flat_predictions, cloned_references):
        try:
            score_results = scorer.score_multi(ref_list, pred)
            aggregator.add_scores(score_results)
        except Exception as e:
            logger.warning(f"Error in ROUGE scoring for pred='{pred}', ref_list='{ref_list}': {e}")
    
    aggregated_scores: dict[str, float] = {rt: np.nan for rt in rouge_types}
    try:
        result = aggregator.aggregate()
        for key in result:
            aggregated_scores[key] = result[key].mid.fmeasure.item()
    except Exception as e:
        logger.warning(f"Error in aggregating ROUGE scores: {e}")
    
    return aggregated_scores, get_gpu_utilization()


def meteor(
    predictions: list[list[str]],
    references: list[list[str]]
) -> tuple[float, int]:
    """Computes the average METEOR score."""
    flat_predictions, cloned_references = preprocess_similarity_inputs(predictions, references)
    if not flat_predictions: return np.nan, get_gpu_utilization()
    
    scores = []
    for pred_str, refs_list_str in zip(flat_predictions, cloned_references):
        tokenized_pred = word_tokenize(pred_str)
        tokenized_refs_list = [word_tokenize(ref_str) for ref_str in refs_list_str]
        
        current_score = np.nan
        try:
            current_score = meteor_score.meteor_score(
                references=tokenized_refs_list,
                hypothesis=tokenized_pred,
                alpha=0.9,
                beta=3,
                gamma=0.5
            )
        except Exception as e:
            logger.warning(f"Error in METEOR for pred='{pred_str}', refs='{refs_list_str}': {e}")
        scores.append(current_score)
        
    return np.nanmean(scores).item() if scores else np.nan, get_gpu_utilization()


def bertscore(
    predictions: list[list[str]],
    references: list[list[str]]
) -> tuple[float, int]:
    """Computes the average BERTScore F1."""
    flat_predictions, cloned_references = preprocess_similarity_inputs(predictions, references)
    if not flat_predictions: return np.nan, get_gpu_utilization()
    
    model_type = bert_score.utils.lang2model['en']
    num_layers = bert_score.utils.model2layers[model_type]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scorer = bert_score.BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        idf=False,
        idf_sents=None,
        device=device,
        lang='en',
        rescale_with_baseline=False,
        use_fast_tokenizer=False,
        baseline_path=None,
    )

    f_measure = np.nan
    try:
        # P, R, F1 (precision, recall, F1)
        _, _, f1_scores = scorer.score(
            cands=flat_predictions, 
            refs=cloned_references, 
            verbose=False,
            batch_size=64
        )
        f_measure = f1_scores.mean().item()
    except Exception as e:
        logger.warning(f"Error in BERTScore calculation: {e}")
        
    return f_measure, get_gpu_utilization()

def character_cer(
    predictions: list[list[str]],
    references: list[list[str]]
) -> tuple[float, int]:
    """Computes the average Character Error Rate (CER)."""
    flat_predictions, cloned_references = preprocess_similarity_inputs(predictions, references)
    if not flat_predictions: return np.nan, get_gpu_utilization()
    tokenized_predictions = [p.split() for p in flat_predictions]
    tokenized_multi_references = [[ref.split() for ref in refs] for refs in cloned_references]

    cer_scores = []
    for pred_str, refs_list_str in zip(tokenized_predictions, tokenized_multi_references):
        min_cer_for_pred = np.inf
        for ref_str in refs_list_str:
            current_cer = np.nan
            try:
                current_cer = calculate_cer(pred_str, ref_str)
            except Exception as e:
                logger.warning(f"Error in CER calculation for pred='{pred_str}', ref='{ref_str}': {e}")
            
            if not np.isnan(current_cer) and current_cer < min_cer_for_pred:
                min_cer_for_pred = current_cer
        
        if min_cer_for_pred != np.inf:
            cer_scores.append(min_cer_for_pred)
            
    return np.nanmean(cer_scores).item() if cer_scores else np.nan, get_gpu_utilization()


def nist_mt(
    predictions: list[list[str]],
    references: list[list[str]]
) -> tuple[float, int]:
    """Computes the corpus-level NIST score."""
    tokenizer = NISTTokenizer()
    nist_scores = []
    for pred_list_idx, pred_list in enumerate(predictions):
        row_references = [references[pred_list_idx] for _ in range(len(pred_list))]
        tokenized_predictions = [
            tokenizer.tokenize(pred, return_str=False, lowercase=False, western_lang=True)
            for pred in pred_list
        ]
        tokenized_references = [
            [
                tokenizer.tokenize(ref, return_str=False, lowercase=False, western_lang=True)
                for ref in ref_set
            ]
            for ref_set in row_references
        ]
    
        score_obtained = False
        current_n_gram = 5 
        score = np.nanprod

    
        while current_n_gram >= 1 and not score_obtained:
            try:
                score = corpus_nist(
                    list_of_references=tokenized_references,
                    hypotheses=tokenized_predictions,
                    n=current_n_gram
                )
                score_obtained = True
            except ZeroDivisionError:
                logger.debug(f"ZeroDivisionError in NIST score with n={current_n_gram}. Trying n={current_n_gram-1}.")
                current_n_gram -= 1
            except Exception as e:
                logger.warning(f"Error in NIST score calculation with n={current_n_gram}: {e}")
                break

        if not score_obtained:
            logger.warning(f"Could not obtain NIST score for the corpus. Returning NaN.")
        
        nist_scores.append(score)
        
    return np.nanmean(nist_scores).item() if nist_scores else np.nan, get_gpu_utilization()


def chrf(
    predictions: list[list[str]],
    references: list[list[str]]
) -> tuple[float, int]:
    """Computes the corpus-level ChrF score."""
    flat_predictions, cloned_references = preprocess_similarity_inputs(predictions, references)
    if not flat_predictions: return np.nan, get_gpu_utilization()
    transformed_references = list(zip(*cloned_references))
    transformed_references = [list(r) for r in transformed_references]
    
    chrf_scorer = sacrebleu.CHRF(
        char_order=sacrebleu.CHRF.CHAR_ORDER,
        word_order=sacrebleu.CHRF.WORD_ORDER,
        beta=sacrebleu.CHRF.BETA,
        lowercase=False,
        whitespace=False, 
        eps_smoothing=False
    )
    score = np.nan
    try:
        result = chrf_scorer.corpus_score(hypotheses=flat_predictions, references=transformed_references)
        score = result.score
    except Exception as e:
        logger.warning(f"Error in CHRF calculation: {e}")
        
    return score, get_gpu_utilization()


def cosine_similarity(
    predictions: list[list[str]],
    references: list[list[str]]
) -> tuple[float, int]:
    """Computes the average cosine similarity using sentence-transformer embeddings."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-mpnet-base-v2", trust_remote_code=True, device=device) 
    model.similarity_fn_name = SimilarityFunction.COSINE
    similarities_all = []
    
    for pred_list_idx, pred_list in enumerate(predictions):
        score = np.nan
        try: 
            pred_embeddings = model.encode(pred_list, convert_to_tensor=True)
            ref_embeddings = model.encode(references[pred_list_idx], convert_to_tensor=True)
            similarities = model.similarity(pred_embeddings, ref_embeddings)
            similarities = similarities.cpu().numpy()
            score = np.mean(similarities).item()
        except Exception as e:
            logger.warning(f"Error in cosine similarity calculation: {e}")
        similarities_all.append(score)
        
    return np.nanmean(similarities_all).item() if similarities_all else np.nan, get_gpu_utilization()


AVAILABLE_METRICS = {
    "sacrebleu": sacrebleu_score,
    "rouge": rouge,
    "meteor": meteor,
    "bertscore": bertscore,
    "cer": character_cer,
    "nist_mt": nist_mt,
    "chrf": chrf,
    "cosine_similarity": cosine_similarity,
}

DERIVED_METRICS_SOURCES = {
    "rouge1": "rouge",
    "rouge2": "rouge",
    "rougeL": "rouge",
    "rougeLsum": "rouge",
}

HIGHER_IS_BETTER = {
    "sacrebleu": True,
    "rouge1": True, "rouge2": True, "rougeL": True, "rougeLsum": True,
    "meteor": True,
    "bertscore": True,
    "cer": False,
    "nist_mt": True,
    "chrf": True,
    "cosine_similarity": True
}


def compute_similarity_scores(
    predictions: list[list[str]],
    references: list[list[str]],
    metric_list: list[str],
) -> dict[str, float]:
    """
    Computes a set of similarity scores between predictions and references.

    Args:
        predictions (list[list[str]]): A list of prediction groups.
        references (list[list[str]]): A list of reference groups.
        metric_list (list[str]): A list of metric names to compute.

    Returns:
        (dict[str, float]): A dictionary mapping metric names to their computed scores.
    """
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and References ({len(references)}) lists must have the same number of samples.")
    
    results_dict: dict[str, float] = {}
    computed_complex_metrics: dict[str, Any] = {}
    gpu_memory_usage: list[int] = []

    base_metrics_to_compute = set()
    for metric_name in metric_list:
        if metric_name in DERIVED_METRICS_SOURCES:
            base_metrics_to_compute.add(DERIVED_METRICS_SOURCES[metric_name])
        elif metric_name in AVAILABLE_METRICS:
            base_metrics_to_compute.add(metric_name)
        else:
            logger.warning(f"Requested metric '{metric_name}' is not available. Skipping.")

    for base_metric_name in tqdm(list(base_metrics_to_compute), desc="Computing similarity metrics"):
        metric_func = AVAILABLE_METRICS[base_metric_name]
        
        score_val, gpu_mem = metric_func(predictions=predictions, references=references)
        gpu_memory_usage.append(gpu_mem)

        if isinstance(score_val, dict):
            computed_complex_metrics[base_metric_name] = score_val
        else:
            results_dict[base_metric_name] = score_val
            
    for metric_name in metric_list:
        if metric_name in DERIVED_METRICS_SOURCES:
            source_metric = DERIVED_METRICS_SOURCES[metric_name]
            if source_metric in computed_complex_metrics and metric_name in computed_complex_metrics[source_metric]:
                results_dict[metric_name] = computed_complex_metrics[source_metric][metric_name]
            else:
                results_dict[metric_name] = np.nan
        elif metric_name not in results_dict and metric_name in AVAILABLE_METRICS :
            pass
        elif metric_name not in AVAILABLE_METRICS and metric_name not in DERIVED_METRICS_SOURCES:
            results_dict[metric_name] = np.nan

    if gpu_memory_usage:
        logger.info(f'Max GPU memory occupied during similarity computation: {np.max(gpu_memory_usage)//1024**2} MB')
    else:
        logger.debug('GPU memory usage not tracked for similarity (no CUDA or no metrics processed).')
        
    return results_dict

