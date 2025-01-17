# Overview

Realtime analytics server for [AI DIAL](https://epam-rail.com). The service consumes the logs stream from [AI DIAL Core](https://github.com/epam/ai-dial-core), analyzes the conversation and writes the analytics to the [InfluxDB](https://www.influxdata.com/).

Refer to [Documentation](https://github.com/epam/ai-dial/blob/main/docs/tutorials/realtime-analytics.md) to learn how to configure AI DAL Core and other necessary components.

## Usage

Check the [AI DIAL Core](https://github.com/epam/ai-dial-core) documentation to configure the way to send the logs to the instance of the realtime analytics server.

The realtime analytics server analyzes the logs stream provided by [Vector](https://vector.dev/docs/reference/configuration/sinks/http/) in the realtime and writes metrics to the InfluxDB.

The logs for `/chat/completions` and `/embeddings` endpoints are saved to the `analytics` measurement with the following tags and fields:

|Tag|Description|
|---|---|
|model| The model name for the request. |
|deployment| The deployment name of the model or application for the request. |
|parent_deployment| The deployment name of the model or application that called the current deployment. |
|execution_path| A list of deployment calls representing the call stack of the request. E.g. `['app1', 'app2', 'model1']` means `app1` called `app2` and `app2` called `model1`. The last element of the list equals to the `deployment` tag. The penultimate element of the list *(when present)* equals to the `parent_deployment` tag. |
|trace_id| OpenTelemetry trace ID. |
|core_span_id| OpenTelemetry span ID generated by DIAL Core. |
|core_parent_span_id| OpenTelemetry span ID generated by DIAL Core that called the span `core_span_id`. |
|project_id| The project ID for the request. |
|language| The language detected for the content of the request. |
|upstream| The upstream endpoint used by the DIAL model. |
|topic| The topic detected for the content of the request. |
|title| The title of the person making the request. |
|response_id| Unique ID of the response. |

|Field|Description|
|---|---|
|user_hash| The unique hash identifying the user. |
|deployment_price| The cost of this specific request, excluding the cost of any requests it directly or indirectly initiated. |
|price| The total cost of the request, including the cost of this request and all related requests it directly or indirectly triggered. |
|number_request_messages| The total number of messages in the request. For chat completion requests it's number of messages in the chat history. For embedding requests it's number of inputs. |
|chat_id| The unique identifier for the conversation that this request is part of. |
|prompt_tokens| The number of tokens in the request. |
|completion_tokens| The number of tokens in the response. |

The logs for the `/rate` endpoint are saved to the `rate_analytics` measurement:

|Tag|Description|
|---|---|
|deployment| The deployment name of the model or application for the request. |
|project_id| The project ID for the request. |
|title| The title of the person making the request. |
|response_id| Unique ID of the response. |
|user_hash| The unique hash identifying the user. |
|chat_id| The unique identifier for the conversation that this request is part of. |

|Field|Description|
|---|---|
|dislike_count| 1 for a thumbs up request, otherwise 0. |
|like_count| 1 for a thumbs down request, otherwise 0. |

## Configuration

Copy `.env.example` to `.env` and customize it for your environment.

### Connection to the InfluxDB

You need to specify the connection options to the InfluxDB instance using the environment variables:

|Variable|Description|
|---|---|
|INFLUX_URL|Url to the InfluxDB to write the analytics data |
|INFLUX_ORG| Name of the InfluxDB organization to write the analytics data |
|INFLUX_BUCKET| Name of the bucket to write the analytics data  |
|INFLUX_API_TOKEN| InfluxDB API Token |

You can follow the [InfluxDB documentation](https://docs.influxdata.com/influxdb/v2/get-started/) to setup InfluxDB locally and acquire the required configuration parameters.

### Other configuration

Also, following environment valuables can be used to configure the service behavior:

|Variable|Default|Description|
|---|---|---|
|MODEL_RATES| {} | Specifies per-token price rates for models in JSON format|
|TOPIC_MODEL| ./topic_model | Specifies the name or path for the topic model. If the model is specified by name, it will be downloaded from, the [Huggingface]( https://huggingface.co/).|
|TOPIC_EMBEDDINGS_MODEL| None | Specifies the name or path for the embeddings model used with the topic model. If the model is specified by name, it will be downloaded from, the [Huggingface]( https://huggingface.co/). If None, the name will be used from the topic model config.|

Example of the MODEL_RATES configuration:

```json
{
    "gpt-4": {
        "unit":"token",
        "prompt_price":"0.00003",
        "completion_price":"0.00006"
    },
    "gpt-35-turbo": {
        "unit":"token",
        "prompt_price":"0.0000015",
        "completion_price":"0.000002"
    },
    "gpt-4-32k": {
        "unit":"token",
        "prompt_price":"0.00006",
        "completion_price":"0.00012"
    },
    "text-embedding-ada-002": {
        "unit":"token",
        "prompt_price":"0.0000001"
    },
    "chat-bison@001": {
        "unit":"char_without_whitespace",
        "prompt_price":"0.0000005",
        "completion_price":"0.0000005"
    }
}
```

## Developer environment

This project uses [Python>=3.11](https://www.python.org/downloads/) and [Poetry>=1.6.1](https://python-poetry.org/) as a dependency manager.
Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```sh
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.

## Build

To build the wheel packages run:

```sh
make build
```

## Run

To run the development server locally run:

```sh
make serve
```

The server will be running as http://localhost:5001

## Docker

To build the docker image run:

```sh
make docker_build
```

To run the server locally from the docker image run:

```sh
make docker_serve
```

The server will be running as http://localhost:5001

## Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

## Test

Run unit tests locally:

```sh
make test
```

## Clean

To remove the virtual environment and build artifacts:

```sh
make clean
```
