while read p;
do
	python3.8 inference_sentiment.py $p
done < kenneth-hf-repos.txt

