# Setup

This tutorial expects two OpenAI-compatible model endpoints and one OpenClaw
Gateway endpoint, all reachable by Tailscale.

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
Gemma4 vLLM:         100.70.253.93:8000
MiMo audio wrapper:  100.70.78.122:19080
MiMo Triton:         100.70.78.122:18000
OpenClaw Gateway:    100.70.78.122:18788
```

The tutorial calls the OpenAI-compatible endpoints:

```text
http://100.70.253.93:8000/v1
http://100.70.78.122:19080/v1
```

OpenClaw is different: it is the agent gateway used for SkillOpt rollouts, not
an OpenAI-compatible `/v1` model server. The local OpenClaw config currently
uses `bind: tailnet` and port `18788`.

To inspect the configured OpenClaw gateway port:

```bash
openclaw config get gateway.port
```

To inspect whether the gateway is listening on the current machine:

```bash
ss -ltnp | rg openclaw
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
