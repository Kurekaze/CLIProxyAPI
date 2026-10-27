package chat_completions

import (
	"context"
	"testing"

	"github.com/tidwall/gjson"
)

func TestGeminiSafetyEvidence(t *testing.T) {
	ratings := `[{"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","probability":"MEDIUM","probabilityScore":0.6,"severity":"HARM_SEVERITY_LOW","severityScore":0.2,"blocked":true}]`
	finishMessage := `"Response was blocked due to SAFETY."`
	raw := []byte(`{"candidates":[{"index":0,"content":{"parts":[{"text":"hi"}]},"finishReason":"SAFETY","safetyRatings":` + ratings + `,"finishMessage":` + finishMessage + `}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}`)

	var param any
	chunks := ConvertGeminiResponseToOpenAI(context.Background(), "gemini", nil, nil, raw, &param)
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks", len(chunks))
	}
	nonStream := ConvertGeminiResponseToOpenAINonStream(context.Background(), "gemini", nil, nil, raw, &param)

	for name, got := range map[string][]byte{"stream": chunks[0], "non-stream": nonStream} {
		if gjson.GetBytes(got, "choices.0.safetyRatings").Raw != ratings {
			t.Fatalf("%s: safetyRatings not preserved: %s", name, got)
		}
		if gjson.GetBytes(got, "choices.0.finishMessage").Raw != finishMessage {
			t.Fatalf("%s: finishMessage not preserved: %s", name, got)
		}
		if gjson.GetBytes(got, "choices.0.finish_reason").String() != "content_filter" {
			t.Fatalf("%s: finish_reason changed: %s", name, got)
		}
	}
}

func TestGeminiPromptFeedbackEvidence(t *testing.T) {
	feedback := `{"blockReason":"PROHIBITED_CONTENT","blockReasonMessage":"blocked"}`
	for _, usage := range []string{"", `,"usageMetadata":{"promptTokenCount":10}`} {
		raw := []byte(`{"promptFeedback":` + feedback + usage + `}`)
		var param any
		chunks := ConvertGeminiResponseToOpenAI(context.Background(), "gemini", nil, nil, raw, &param)
		if len(chunks) != 1 || gjson.GetBytes(chunks[0], "promptFeedback").Raw != feedback {
			t.Fatalf("usage=%q: stream promptFeedback not preserved: %q", usage, chunks)
		}
		nonStream := ConvertGeminiResponseToOpenAINonStream(context.Background(), "gemini", nil, nil, raw, &param)
		if gjson.GetBytes(nonStream, "promptFeedback").Raw != feedback {
			t.Fatalf("usage=%q: non-stream promptFeedback not preserved: %s", usage, nonStream)
		}
	}
}

func TestGeminiSafetyEvidenceAbsent(t *testing.T) {
	raw := []byte(`{"candidates":[{"index":0,"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10}}`)
	var param any
	got := ConvertGeminiResponseToOpenAI(context.Background(), "gemini", nil, nil, raw, &param)[0]
	for _, path := range []string{"choices.0.safetyRatings", "choices.0.finishMessage", "promptFeedback"} {
		if gjson.GetBytes(got, path).Exists() {
			t.Fatalf("invented %s: %s", path, got)
		}
	}
}
