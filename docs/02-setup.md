# Setup

This tutorial expects two OpenAI-compatible endpoints, both reachable by Tailscale.

Gemma4 vLLM:

```bash
curl http://100.70.253.93:8000/v1/models
```

MiMo S2S wrapper:

```bash
curl http://100.70.78.122:19080/v1/models
```

Current service map:

```text
Gemma4 vLLM:       100.70.253.93:8000
MiMo audio wrapper: 100.70.78.122:19080
MiMo Triton:        100.70.78.122:18000
```

The tutorial calls the OpenAI-compatible endpoints:

```text
http://100.70.253.93:8000/v1
http://100.70.78.122:19080/v1
```

To see the Tailscale IP of the current machine:

```bash
tailscale ip -4
```

Alternative:

```bash
ip -4 addr show tailscale0
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```
