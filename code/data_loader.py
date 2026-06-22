import random

import ir_datasets
from config import DATASET_NAME, MAX_DOCS
from utils import setup_logger

logger = setup_logger("DataLoader")

# قائمة المجالات المتاحة في cqadupstack
CQADUPSTACK_DOMAINS = [
    'beir/cqadupstack/android',
    'beir/cqadupstack/english',
    'beir/cqadupstack/gaming',
    'beir/cqadupstack/gis',
    'beir/cqadupstack/mathematica',
    'beir/cqadupstack/physics',
    'beir/cqadupstack/programmers',
    'beir/cqadupstack/stats',
    'beir/cqadupstack/tex',
    'beir/cqadupstack/unix',
    'beir/cqadupstack/webmasters',
    'beir/cqadupstack/wordpress',
]

def load_dataset(dataset_name):
    """
    تحميل قاعدة بيانات واحدة
    """
    logger.info(f"جاري تحميل قاعدة البيانات: {dataset_name}...")
    try:
        dataset = ir_datasets.load(dataset_name)
        logger.info("تم تحميل قاعدة البيانات بنجاح.")
        return dataset
    except Exception as e:
        logger.error(f"حدث خطأ أثناء تحميل قاعدة البيانات: {e}")
        raise

def load_multiple_domains(domains=None):
    """
    تحميل عدة مجالات من cqadupstack ودمجها معاً
    
    Args:
        domains: قائمة المجالات المراد تحميلها
                 إذا كانت None، سيتم تحميل جميع المجالات
    
    Returns:
        tuple: (docs, queries, qrels) مدمجة من جميع المجالات
    """
    if domains is None:
        domains = CQADUPSTACK_DOMAINS
    
    logger.info(f"جاري تحميل {len(domains)} مجال من cqadupstack...")
    
    all_docs = {}
    all_queries = {}
    all_qrels = {}
    
    for i, domain in enumerate(domains):
        logger.info(f"\n[{i+1}/{len(domains)}] تحميل: {domain}")
        
        try:
            dataset = load_dataset(domain)
            
            # تحميل الوثائق
            logger.info(f"  - جاري استخراج الوثائق...")
            domain_docs = {}
            for doc in dataset.docs_iter():
                # إضافة اسم المجال إلى معرف الوثيقة لتجنب التضارب
                unique_doc_id = f"{domain.split('/')[-1]}_{doc.doc_id}"
                domain_docs[unique_doc_id] = doc.text
            
            logger.info(f"    ✓ تم استخراج {len(domain_docs)} وثيقة")
            all_docs.update(domain_docs)
            
            # تحميل الاستعلامات
            logger.info(f"  - جاري استخراج الاستعلامات...")
            domain_queries = {}
            for query in dataset.queries_iter():
                domain_queries[query.query_id] = query.text
            
            logger.info(f"    ✓ تم استخراج {len(domain_queries)} استعلام")
            all_queries.update(domain_queries)
            
            # تحميل qrels
            logger.info(f"  - جاري استخراج qrels...")
            domain_qrels = {}
            for qrel in dataset.qrels_iter():
                if qrel.query_id not in domain_qrels:
                    domain_qrels[qrel.query_id] = {}
                
                # استخدام نفس معرف الوثيقة المعدل
                unique_doc_id = f"{domain.split('/')[-1]}_{qrel.doc_id}"
                domain_qrels[qrel.query_id][unique_doc_id] = qrel.relevance
            
            logger.info(f"    ✓ تم استخراج {len(domain_qrels)} استعلام مقيم")
            
            # دمج qrels
            for query_id, docs in domain_qrels.items():
                if query_id not in all_qrels:
                    all_qrels[query_id] = {}
                all_qrels[query_id].update(docs)
                
        except Exception as e:
            logger.warning(f"  ❌ خطأ في تحميل {domain}: {e}")
            continue
    
    logger.info(f"\n✅ اكتمل تحميل جميع المجالات!")
    logger.info(f"   - إجمالي الوثائق: {len(all_docs)}")
    logger.info(f"   - إجمالي الاستعلامات: {len(all_queries)}")
    logger.info(f"   - إجمالي الاستعلامات المقيمة: {len(all_qrels)}")
    
    return all_docs, all_queries, all_qrels

def get_documents(dataset, max_docs=MAX_DOCS):
    """
    استخراج الوثائق من قاعدة بيانات واحدة
    """
    logger.info("جاري استخراج الوثائق...")
    docs = {}
    
    for i, doc in enumerate(dataset.docs_iter()):
        if max_docs and i >= max_docs:
            break
        docs[doc.doc_id] = doc.text
        
        if (i + 1) % 10000 == 0:
            logger.info(f"تم استخراج {i + 1} وثيقة...")
            
    logger.info(f"اكتمل استخراج الوثائق. العدد الإجمالي: {len(docs)}")
    return docs

def load_or_build_processed_corpus(
    document_store,
    preprocessor,
    dataset=None,
    max_docs=MAX_DOCS,
    dataset_name=None,
):
    """
    Load originals + processed tokens from SQLite when cached, otherwise build,
    persist, and return both collections.

    Returns:
        (docs, processed_docs, documents_source, processed_source)
        sources are "sqlite" or "ir_datasets"/"computed"
    """
    dataset_name = dataset_name or DATASET_NAME

    if document_store is not None:
        bundle = document_store.load_corpus_bundle(
            dataset_name, max_docs, preprocessor
        )
        if bundle is not None:
            return bundle[0], bundle[1], "sqlite", "sqlite"

    docs = None
    documents_source = "ir_datasets"

    if document_store is not None:
        docs = document_store.load_corpus(dataset_name, max_docs)
        if docs is not None:
            documents_source = "sqlite"

    if docs is None:
        if dataset is None:
            dataset = load_dataset(dataset_name)
        docs = get_documents(dataset, max_docs=max_docs)
        documents_source = "ir_datasets"

    processed_docs = preprocessor.process_collection(docs)
    processed_source = "computed"

    if document_store is not None:
        document_store.save_corpus_bundle(
            docs,
            processed_docs,
            dataset_name=dataset_name,
            max_docs=max_docs,
            preprocessor=preprocessor,
        )
        if documents_source == "sqlite":
            processed_source = "sqlite"

    return docs, processed_docs, documents_source, processed_source

def load_and_persist_documents(
    dataset,
    max_docs=MAX_DOCS,
    document_store=None,
    dataset_name=None,
):
    """
    Load documents for indexing. Uses SQLite when the same corpus is already stored,
    otherwise reads from ir_datasets and persists originals.
    """
    dataset_name = dataset_name or DATASET_NAME

    if document_store is not None:
        cached_docs = document_store.load_corpus(dataset_name, max_docs)
        if cached_docs is not None:
            return cached_docs

    docs = get_documents(dataset, max_docs=max_docs)
    if document_store is not None:
        document_store.save_corpus(
            docs,
            dataset_name=dataset_name,
            max_docs=max_docs,
        )
    return docs

def get_queries(dataset):
    """
    استخراج الاستعلامات من قاعدة بيانات واحدة
    """
    logger.info("جاري استخراج الاستعلامات...")
    queries = {}
    
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
        
    logger.info(f"اكتمل استخراج الاستعلامات. العدد الإجمالي: {len(queries)}")
    return queries

def get_qrels(dataset):
    """
    استخراج تقييمات الصلة من قاعدة بيانات واحدة
    """
    if not hasattr(dataset, 'qrels_iter'):
        raise AttributeError(
            "هذه القاعدة لا تحتوي على qrels. "
            "في ir_datasets استخدم beir/quora/test أو beir/quora/dev بدلاً من beir/quora."
        )

    logger.info("جاري استخراج تقييمات الصلة (qrels)...")
    qrels = {}
    
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {}
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
    logger.info(f"اكتمل استخراج تقييمات الصلة. عدد الاستعلامات المقيمة: {len(qrels)}")
    return qrels

def select_eval_queries(queries, qrels, max_queries=None, random_seed=42):
    """
    Select queries that have qrels for evaluation.

    Args:
        queries: {query_id: text}
        qrels: {query_id: {doc_id: relevance}}
        max_queries: if set, randomly sample this many qrel-covered queries
        random_seed: seed for random sampling

    Returns:
        {query_id: text} subset for evaluation
    """
    covered_ids = [qid for qid in qrels if qid in queries]
    logger.info(
        f"Queries: {len(queries)} total, {len(covered_ids)} with qrels."
    )

    if max_queries is not None and max_queries < len(covered_ids):
        random.seed(random_seed)
        selected_ids = random.sample(covered_ids, max_queries)
        logger.info(f"Randomly sampled {max_queries} queries (seed={random_seed}).")
    else:
        selected_ids = covered_ids
        logger.info(f"Using all {len(selected_ids)} qrel-covered queries.")

    return {qid: queries[qid] for qid in selected_ids}

if __name__ == "__main__":
    # اختبار: تحميل أول 3 مجالات فقط
    test_domains = CQADUPSTACK_DOMAINS[:3]
    docs, queries, qrels = load_multiple_domains(domains=test_domains)
    print(f"\n✅ تم تحميل {len(docs)} وثيقة من {len(test_domains)} مجالات")
