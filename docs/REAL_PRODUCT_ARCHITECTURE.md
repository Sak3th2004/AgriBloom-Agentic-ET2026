# AgriBloom real-product architecture

AgriBloom should not be a demo dashboard. It should be one advanced advisory
backend exposed through lightweight farmer access channels.

## Product surfaces

1. **PWA web app**
   - Best for rich camera flow and result cards.
   - Works in a browser and can be installed on low-end phones.
   - No frontend build step or heavy JavaScript framework.

2. **WhatsApp bot**
   - Best for reach and daily farmer behavior.
   - Accepts text and image messages through the WhatsApp Cloud API webhook.
   - Replies with short farmer-readable advice.

3. **Mobile app**
   - Useful later for deeper offline mode and local model inference.
   - Should reuse the same backend contract.

## Backend contract

Every channel should consume the same `farmer_advice` object:

- crop
- problem
- risk level
- confidence
- what to do today
- treatment guidance
- what not to do
- when to call an expert
- helpline
- advanced technical details

This prevents each frontend from parsing long generated text.

## Advanced fallback stack

The system should use a layered approach:

1. local trained vision model for known crop/disease classes
2. vision-language model fallback for unknown crops or low confidence
3. RAG/knowledge base for treatment grounding
4. deterministic compliance guardrails
5. structured farmer output

Cloud LLM providers are optional and configured through environment variables.
The app should continue to work without one provider by falling back to another.

## Safe self-learning loop

AgriBloom should not automatically retrain from raw farmer feedback. That can
make the model worse or unsafe.

The safe loop is:

1. collect farmer feedback and corrections
2. store as JSONL review queue
3. review and clean data
4. use approved feedback for retrieval/evaluation/fine-tuning
5. redeploy improved models/prompts

Current implementation records feedback through `/api/feedback` and WhatsApp
message metadata for review.

## Run

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

