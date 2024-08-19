from fastapi import FastAPI, Request, Response


app = FastAPI()


@app.get("/")
def default_route(request: Request) -> Response:
    return Response(status_code=200, content="Root")
