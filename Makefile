PATH_TO_TESTSET := data/raw/test.csv
PATH_TO_RESULTS := results.csv

build:
	docker build . --tag text_clf_service

run-service:
	docker run --publish 8000:8000 text_clf_service

get-prediction:
	curl  -X GET "http://127.0.0.1:8000/predict?text=123"

get-test-predictions:
	curl --data-binary @$(PATH_TO_TESTSET) -H "Content-Type: text/csv"  http://127.0.0.1:8000/test_predictions > $(PATH_TO_RESULTS)

# for dev
update-conda-env:
	 conda env export --from-history > env_dev.yml

test-docker:
	docker run --rm -it --entrypoint /bin/bash text_clf_service
