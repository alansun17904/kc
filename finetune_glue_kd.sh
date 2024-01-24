python3 regularizer/train_glue.py mnli "bert-base-uncased" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 3
python3 regularizer/train_glue.py mnli "t5-base" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 3 -encoder_decoder True
python3 regularizer/train_glue.py mnli "gpt2" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 3
# python3 regularizer/train_glue.py cola "bert-base-uncased" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50 
# python3 regularizer/train_glue.py cola "t5-base" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50 -encoder_decoder True
# python3 regularizer/train_glue.py cola "gpt2" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50 
# python3 regularizer/train_glue.py mrpc "bert-base-uncased" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50 
# python3 regularizer/train_glue.py mrpc "t5-base" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50 -encoder_decoder True
# python3 regularizer/train_glue.py mrpc "gpt2" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50 
python3 regularizer/train_glue.py qnli "bert-base-uncased" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 5 
python3 regularizer/train_glue.py qnli "t5-base" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 5 -encoder_decoder True
python3 regularizer/train_glue.py qnli "gpt2" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 5
python3 regularizer/train_glue.py qqp "bert-base-uncased" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 3
python3 regularizer/train_glue.py qqp "t5-base" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 3 -encoder_decoder True
python3 regularizer/train_glue.py qqp "gpt2" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 3
python3 regularizer/train_glue.py rte "bert-base-uncased" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50 
python3 regularizer/train_glue.py rte "t5-base" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50 -encoder_decoder True
python3 regularizer/train_glue.py rte "gpt2" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 50
# python3 regularizer/train_glue.py wnli "bert-base-uncased" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 100
# python3 regularizer/train_glue.py wnli "t5-base" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 100 -encoder_decoder True
# python3 regularizer/train_glue.py wnli "gpt2" 2 1 "2e-2" "1e-2" "5e-5" "1e-9" -epochs 100








