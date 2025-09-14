python-run:
	fastapi dev app.py

build:
	docker build -t cat-classifier .

run: build
	docker run -p 8080:8080 cat-classifier 

clean:
	docker stop cat-classifier