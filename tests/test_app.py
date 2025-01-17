import json
import re

from fastapi.testclient import TestClient

import aidial_analytics_realtime.app as app
from tests.mocks import InfluxWriterMock, TestTopicModel


def test_chat_completion_plain_text():
    write_api_mock = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
                            "time": "2023-08-16T19:42:39.997",
                            "body": json.dumps(
                                {
                                    "messages": [
                                        {"role": "system", "content": ""},
                                        {"role": "user", "content": "ping"},
                                    ],
                                    "model": "gpt-4",
                                    "max_tokens": 2000,
                                    "stream": True,
                                    "n": 1,
                                    "temperature": 0.0,
                                }
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1692214960,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":null}]}\n\ndata: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1692214960,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":189,"prompt_tokens":22,"total_tokens":211}}\n\ndata: [DONE]\n',
                        },
                    }
                )
            },
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-2"},
                        "project": {"id": "PROJECT-KEY-2"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/gpt-4/chat/completions",
                            "time": "2023-11-24T03:33:40.39",
                            "body": json.dumps(
                                {
                                    "messages": [
                                        {"role": "system", "content": ""},
                                        {"role": "user", "content": "ping"},
                                    ],
                                    "model": "gpt-4",
                                    "max_tokens": 2000,
                                    "stream": True,
                                    "n": 1,
                                    "temperature": 0.0,
                                }
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": 'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1700828102,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"po"},"finish_reason":null}]}\n\ndata: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1700828102,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":null}]}\n\ndata: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1700828102,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":189,"prompt_tokens":22,"total_tokens":211}}\n\ndata: [DONE]\n',
                        },
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert write_api_mock.points == [
        'analytics,core_parent_span_id=undefined,core_span_id=undefined,deployment=gpt-4,execution_path=undefined,language=undefined,model=gpt-4,parent_deployment=undefined,project_id=PROJECT-KEY,response_id=chatcmpl-1,title=undefined,topic=ping\\n\\npong,trace_id=undefined,upstream=undefined chat_id="chat-1",completion_tokens=189i,deployment_price=0,number_request_messages=2i,price=0,prompt_tokens=22i,user_hash="undefined" 1692214959997000000',
        'analytics,core_parent_span_id=undefined,core_span_id=undefined,deployment=gpt-4,execution_path=undefined,language=undefined,model=gpt-4,parent_deployment=undefined,project_id=PROJECT-KEY-2,response_id=chatcmpl-2,title=undefined,topic=ping\\n\\npong,trace_id=undefined,upstream=undefined chat_id="chat-2",completion_tokens=189i,deployment_price=0,number_request_messages=2i,price=0,prompt_tokens=22i,user_hash="undefined" 1700796820390000000',
    ]


def test_chat_completion_plain_text_no_body():
    write_api_mock = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "token_usage": {
                            "completion_tokens": 189,
                            "prompt_tokens": 22,
                            "total_tokens": 211,
                            "deployment_price": 0.001,
                            "price": 0.001,
                        },
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
                            "time": "2023-08-16T19:42:39.997",
                        },
                        "response": {"status": "200"},
                    }
                )
            },
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-2"},
                        "project": {"id": "PROJECT-KEY-2"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "token_usage": {
                            "completion_tokens": 189,
                            "prompt_tokens": 22,
                            "total_tokens": 211,
                            "deployment_price": 0.001,
                            "price": 0.001,
                        },
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/gpt-4/chat/completions",
                            "time": "2023-11-24T03:33:40.39",
                        },
                        "response": {"status": "200"},
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert len(write_api_mock.points) == 2

    assert re.match(
        r'analytics,core_parent_span_id=undefined,core_span_id=undefined,deployment=gpt-4,execution_path=undefined,language=undefined,model=gpt-4,parent_deployment=undefined,project_id=PROJECT-KEY,response_id=(.+?),title=undefined,trace_id=undefined,upstream=undefined chat_id="chat-1",completion_tokens=189i,deployment_price=0.001,number_request_messages=0i,price=0.001,prompt_tokens=22i,user_hash="undefined" 1692214959997000000',
        write_api_mock.points[0],
    )

    assert re.match(
        r'analytics,core_parent_span_id=undefined,core_span_id=undefined,deployment=gpt-4,execution_path=undefined,language=undefined,model=gpt-4,parent_deployment=undefined,project_id=PROJECT-KEY-2,response_id=(.+?),title=undefined,trace_id=undefined,upstream=undefined chat_id="chat-2",completion_tokens=189i,deployment_price=0.001,number_request_messages=0i,price=0.001,prompt_tokens=22i,user_hash="undefined" 1700796820390000000',
        write_api_mock.points[1],
    )


def test_chat_completion_list_content():
    write_api_mock = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
                            "time": "2023-08-16T19:42:39.997",
                            "body": json.dumps(
                                {
                                    "messages": [
                                        {
                                            "role": "system",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "act as a helpful assistant",
                                                }
                                            ],
                                        },
                                        {"role": "user", "content": "ping"},
                                    ],
                                    "model": "gpt-4",
                                    "max_tokens": 2000,
                                    "stream": True,
                                    "n": 1,
                                    "temperature": 0.0,
                                }
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1692214960,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":null}]}\n\ndata: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1692214960,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":189,"prompt_tokens":22,"total_tokens":211}}\n\ndata: [DONE]\n',
                        },
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert write_api_mock.points == [
        'analytics,core_parent_span_id=undefined,core_span_id=undefined,deployment=gpt-4,execution_path=undefined,language=undefined,model=gpt-4,parent_deployment=undefined,project_id=PROJECT-KEY,response_id=chatcmpl-1,title=undefined,topic=act\\ as\\ a\\ helpful\\ assistant\\n\\nping\\n\\npong,trace_id=undefined,upstream=undefined chat_id="chat-1",completion_tokens=189i,deployment_price=0,number_request_messages=2i,price=0,prompt_tokens=22i,user_hash="undefined" 1692214959997000000',
    ]


def test_chat_completion_none_content():
    write_api_mock = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
                            "time": "2023-08-16T19:42:39.997",
                            "body": json.dumps(
                                {
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": "what's the weather like?",
                                        },
                                        {
                                            "role": "assistant",
                                            "tool_calls": [
                                                {
                                                    "id": "xyz",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "get_weather",
                                                        "arguments": {},
                                                    },
                                                }
                                            ],
                                        },
                                        {
                                            "role": "tool",
                                            "id": "xyz",
                                            "content": "It's sunny today.",
                                        },
                                        {
                                            "role": "user",
                                            "content": "2+3=?",
                                        },
                                    ],
                                    "model": "gpt-4",
                                    "max_tokens": 2000,
                                    "stream": True,
                                    "n": 1,
                                    "temperature": 0.0,
                                }
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1692214960,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"5"},"finish_reason":null}]}\n\ndata: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1692214960,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":189,"prompt_tokens":22,"total_tokens":211}}\n\ndata: [DONE]\n',
                        },
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert write_api_mock.points == [
        'analytics,core_parent_span_id=undefined,core_span_id=undefined,deployment=gpt-4,execution_path=undefined,language=undefined,model=gpt-4,parent_deployment=undefined,project_id=PROJECT-KEY,response_id=chatcmpl-1,title=undefined,topic=what\'s\\ the\\ weather\\ like?\\n\\nIt\'s\\ sunny\\ today.\\n\\n2+3\\=?\\n\\n5,trace_id=undefined,upstream=undefined chat_id="chat-1",completion_tokens=189i,deployment_price=0,number_request_messages=4i,price=0,prompt_tokens=22i,user_hash="undefined" 1692214959997000000',
    ]


def test_embeddings_plain_text():
    write_api_mock: app.InfluxWriterAsync = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "text-embedding-3-small",
                        "token_usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 2,
                            "total_tokens": 2,
                            "deployment_price": 0.001,
                            "price": 0.001,
                        },
                        "parent_deployment": "assistant",
                        "trace": {
                            "trace_id": "5dca3d6ed5d22b6ab574f27a6ab5ec14",
                            "core_span_id": "9ade2b6fef0a716d",
                            "core_parent_span_id": "20e7e64715abbe97",
                        },
                        "execution_path": [None, "b", "c"],
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-03-15-preview",
                            "time": "2023-08-16T19:42:39.997",
                            "body": json.dumps({"input": ["fish", "cat"]}),
                        },
                        "response": {
                            "status": "200",
                            "body": json.dumps(
                                {
                                    "data": [
                                        {
                                            "embedding": [0.1, 0.2],
                                            "index": 0,
                                            "object": "embedding",
                                        },
                                        {
                                            "embedding": [0.3, 0.4],
                                            "index": 1,
                                            "object": "embedding",
                                        },
                                    ],
                                    "model": "text-embedding-3-small",
                                    "object": "list",
                                    "usage": {
                                        "prompt_tokens": 43,
                                        "total_tokens": 43,
                                    },
                                }
                            ),
                        },
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert len(write_api_mock.points) == 1
    assert re.match(
        r'analytics,core_parent_span_id=20e7e64715abbe97,core_span_id=9ade2b6fef0a716d,deployment=text-embedding-3-small,execution_path=undefined/b/c,language=undefined,model=text-embedding-3-small,parent_deployment=assistant,project_id=PROJECT-KEY,response_id=(.+?),title=undefined,topic=fish\\n\\ncat,trace_id=5dca3d6ed5d22b6ab574f27a6ab5ec14,upstream=undefined chat_id="chat-1",completion_tokens=0i,deployment_price=0.001,number_request_messages=2i,price=0.001,prompt_tokens=2i,user_hash="undefined" 1692214959997000000',
        write_api_mock.points[0],
    )


def test_embeddings_no_body():
    write_api_mock: app.InfluxWriterAsync = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "text-embedding-3-small",
                        "token_usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 2,
                            "total_tokens": 2,
                            "deployment_price": 0.001,
                            "price": 0.001,
                        },
                        "parent_deployment": "assistant",
                        "trace": {
                            "trace_id": "5dca3d6ed5d22b6ab574f27a6ab5ec14",
                            "core_span_id": "9ade2b6fef0a716d",
                            "core_parent_span_id": "20e7e64715abbe97",
                        },
                        "execution_path": [None, "b", "c"],
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-03-15-preview",
                            "time": "2023-08-16T19:42:39.997",
                        },
                        "response": {"status": "200"},
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert len(write_api_mock.points) == 1
    assert re.match(
        r'analytics,core_parent_span_id=20e7e64715abbe97,core_span_id=9ade2b6fef0a716d,deployment=text-embedding-3-small,execution_path=undefined/b/c,language=undefined,model=text-embedding-3-small,parent_deployment=assistant,project_id=PROJECT-KEY,response_id=(.+?),title=undefined,trace_id=5dca3d6ed5d22b6ab574f27a6ab5ec14,upstream=undefined chat_id="chat-1",completion_tokens=0i,deployment_price=0.001,number_request_messages=0i,price=0.001,prompt_tokens=2i,user_hash="undefined" 1692214959997000000',
        write_api_mock.points[0],
    )


def test_embeddings_tokens():
    write_api_mock: app.InfluxWriterAsync = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "text-embedding-3-small",
                        "token_usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 2,
                            "total_tokens": 2,
                            "deployment_price": 0.001,
                            "price": 0.001,
                        },
                        "parent_deployment": "assistant",
                        "trace": {
                            "trace_id": "5dca3d6ed5d22b6ab574f27a6ab5ec14",
                            "core_span_id": "9ade2b6fef0a716d",
                            "core_parent_span_id": "20e7e64715abbe97",
                        },
                        "execution_path": [None, "b", "c"],
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-03-15-preview",
                            "time": "2023-08-16T19:42:39.997",
                            "body": json.dumps(
                                {"input": [[1, 3, 4, 5], [6, 7, 8, 9]]}
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": json.dumps(
                                {
                                    "data": [
                                        {
                                            "embedding": [0.1, 0.2],
                                            "index": 0,
                                            "object": "embedding",
                                        },
                                        {
                                            "embedding": [0.3, 0.4],
                                            "index": 1,
                                            "object": "embedding",
                                        },
                                    ],
                                    "model": "text-embedding-3-small",
                                    "object": "list",
                                    "usage": {
                                        "prompt_tokens": 43,
                                        "total_tokens": 43,
                                    },
                                }
                            ),
                        },
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert len(write_api_mock.points) == 1
    assert re.match(
        r'analytics,core_parent_span_id=20e7e64715abbe97,core_span_id=9ade2b6fef0a716d,deployment=text-embedding-3-small,execution_path=undefined/b/c,language=undefined,model=text-embedding-3-small,parent_deployment=assistant,project_id=PROJECT-KEY,response_id=(.+?),title=undefined,topic=undefined,trace_id=5dca3d6ed5d22b6ab574f27a6ab5ec14,upstream=undefined chat_id="chat-1",completion_tokens=0i,deployment_price=0.001,number_request_messages=2i,price=0.001,prompt_tokens=2i,user_hash="undefined" 1692214959997000000',
        write_api_mock.points[0],
    )


def test_data_request_with_new_format():
    write_api_mock: app.InfluxWriterAsync = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "token_usage": {
                            "completion_tokens": 40,
                            "prompt_tokens": 30,
                            "deployment_price": 0.001,
                            "price": 0.001,
                        },
                        "parent_deployment": "assistant",
                        "trace": {
                            "trace_id": "5dca3d6ed5d22b6ab574f27a6ab5ec14",
                            "core_span_id": "9ade2b6fef0a716d",
                            "core_parent_span_id": "20e7e64715abbe97",
                        },
                        "execution_path": [None, "b", "c"],
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
                            "time": "2023-08-16T19:42:39.997",
                            "body": json.dumps(
                                {
                                    "messages": [
                                        {"role": "system", "content": ""},
                                        {"role": "user", "content": "ping"},
                                    ],
                                    "model": "gpt-4",
                                    "max_tokens": 2000,
                                    "stream": True,
                                    "n": 1,
                                    "temperature": 0.0,
                                }
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1692214960,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":null}]}\n\ndata: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1692214960,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":189,"prompt_tokens":22,"total_tokens":211}}\n\ndata: [DONE]\n',
                        },
                    }
                )
            },
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-2"},
                        "project": {"id": "PROJECT-KEY-2"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "token_usage": {
                            "completion_tokens": 40,
                            "prompt_tokens": 30,
                            "price": 0.005,
                        },
                        "trace": {
                            "trace_id": "5dca3d6ed5d22b6ab574f27a6ab5ec14",
                            "core_span_id": "20e7e64715abbe97",
                        },
                        "execution_path": ["a", "b", "c"],
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/openai/deployments/gpt-4/chat/completions",
                            "time": "2023-11-24T03:33:40.39",
                            "body": json.dumps(
                                {
                                    "messages": [
                                        {"role": "system", "content": ""},
                                        {"role": "user", "content": "ping"},
                                    ],
                                    "model": "gpt-4",
                                    "max_tokens": 2000,
                                    "stream": True,
                                    "n": 1,
                                    "temperature": 0.0,
                                }
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": 'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1700828102,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":null}]}\n\ndata: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1700828102,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":189,"prompt_tokens":22,"total_tokens":211}}\n\ndata: [DONE]\n',
                        },
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert write_api_mock.points == [
        'analytics,core_parent_span_id=20e7e64715abbe97,core_span_id=9ade2b6fef0a716d,deployment=gpt-4,execution_path=undefined/b/c,language=undefined,model=gpt-4,parent_deployment=assistant,project_id=PROJECT-KEY,response_id=chatcmpl-1,title=undefined,topic=ping\\n\\npong,trace_id=5dca3d6ed5d22b6ab574f27a6ab5ec14,upstream=undefined chat_id="chat-1",completion_tokens=40i,deployment_price=0.001,number_request_messages=2i,price=0.001,prompt_tokens=30i,user_hash="undefined" 1692214959997000000',
        'analytics,core_parent_span_id=undefined,core_span_id=20e7e64715abbe97,deployment=gpt-4,execution_path=a/b/c,language=undefined,model=gpt-4,parent_deployment=undefined,project_id=PROJECT-KEY-2,response_id=chatcmpl-2,title=undefined,topic=ping\\n\\npong,trace_id=5dca3d6ed5d22b6ab574f27a6ab5ec14,upstream=undefined chat_id="chat-2",completion_tokens=40i,deployment_price=0,number_request_messages=2i,price=0.005,prompt_tokens=30i,user_hash="undefined" 1700796820390000000',
    ]


def test_rate_request():
    write_api_mock = InfluxWriterMock()
    app.app.dependency_overrides[app.InfluxWriterAsync] = lambda: write_api_mock
    app.app.dependency_overrides[app.TopicModel] = lambda: TestTopicModel()

    client = TestClient(app.app)
    response = client.post(
        "/data",
        json=[
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/v1/gpt-4/rate",
                            "time": "2023-08-16T19:42:39.997",
                            "body": json.dumps(
                                {
                                    "responseId": "response_123",
                                    "rate": True,
                                }
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": "",
                        },
                    }
                )
            },
            {
                "message": json.dumps(
                    {
                        "apiType": "DialOpenAI",
                        "chat": {"id": "chat-1"},
                        "project": {"id": "PROJECT-KEY"},
                        "user": {"id": "", "title": ""},
                        "deployment": "gpt-4",
                        "request": {
                            "protocol": "HTTP/1.1",
                            "method": "POST",
                            "uri": "/v1/gpt-4/rate",
                            "time": "2023-11-24T03:33:40.39",
                            "body": json.dumps(
                                {
                                    "responseId": "response_124",
                                    "rate": False,
                                }
                            ),
                        },
                        "response": {
                            "status": "200",
                            "body": "",
                        },
                    }
                )
            },
        ],
    )
    assert response.status_code == 200
    assert write_api_mock.points == [
        "rate_analytics,chat_id=chat-1,deployment=gpt-4,project_id=PROJECT-KEY,response_id=response_123,title=undefined,user_hash=undefined dislike_count=0i,like_count=1i 1692214959997000000",
        "rate_analytics,chat_id=chat-1,deployment=gpt-4,project_id=PROJECT-KEY,response_id=response_124,title=undefined,user_hash=undefined dislike_count=1i,like_count=0i 1700796820390000000",
    ]
