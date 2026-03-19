
BENCHMARK_DATA=benchmark/WebMainBench_100.jsonl
RESULT_DIR=benchmark_results
mkdir $RESULT_DIR

# MinerU-HTML Extractors (Use GPU)
EXTRACTORS=(
    "mineru_html_fallback-html-md"
    "mineru_html-html-md"
)
MODEL_PATH=YOUR_MINERUHTML_MODEL_PATH

for extractor in ${EXTRACTORS[@]}; do
    python eval_baselines.py --bench $BENCHMARK_DATA --task_dir $RESULT_DIR/$extractor --extractor_name  $extractor --model_path $MODEL_PATH --default_config gpu
done



# CPU Extractors
EXTRACTORS=(
"magichtml-html-md"
"readability-html-md"
"trafilatura-html-md"
"resiliparse-text"
"trafilatura-md"
"trafilatura-text"
"fullpage-html-md"
"boilerpy3-text"
"gne-html-md"
"newsplease-text"
"justtext-text"
"boilerpy3-html-md"
"goose3-text"
)

for extractor in ${EXTRACTORS[@]}; do
    python eval_baselines.py --bench $BENCHMARK_DATA --task_dir $RESULT_DIR/$extractor --extractor_name $extractor
done



# ReaderLM Extractors (Use GPU)
extractor=readerlm-text
MODEL_PATH=YOUR_READERLM_MODEL_PATH
python eval_baselines.py --bench $BENCHMARK_DATA --task_dir $RESULT_DIR/$extractor --extractor_name  $extractor --model_path $MODEL_PATH --default_config gpu
