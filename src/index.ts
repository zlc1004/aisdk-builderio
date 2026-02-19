import {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3StreamPart,
  LanguageModelV3FinishReason,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
  APICallError,
} from "@ai-sdk/provider";
import { generateId, loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils";

export interface BuilderProviderSettings {
  baseURL?: string;
  apiKey: string;
  userId: string;
  privateKey?: string;
}

export interface BuilderChatSettings {
  temperature?: number;
  maxOutputTokens?: number;
  topP?: number;
  stopSequences?: string[];
}

export interface BuilderProvider {
  (modelId: string, settings?: BuilderChatSettings): BuilderChatLanguageModel;
  languageModel(modelId: string, settings?: BuilderChatSettings): BuilderChatLanguageModel;
}

interface BuilderCompletionMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface BuilderSSEEvent {
  type: string;
  content?: string;
  delta?: string;
  id?: string;
  stopReason?: string;
  actions?: Array<{ type: string; content: string; id?: string }>;
  messageIndex?: number;
  creditsUsed?: number;
  model?: string;
  user?: {
    source?: string;
    role?: string;
    userId?: string;
  };
  synthetic?: boolean;
}

export class BuilderChatLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3";
  readonly provider = "builder";
  readonly modelId: string;

  private readonly settings: BuilderChatSettings;
  private readonly config: BuilderConfig;

  constructor(
    modelId: string,
    settings: BuilderChatSettings,
    config: BuilderConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  get supportedUrls() {
    return {};
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const messages = this.convertMessages(options.prompt);
    const url = this.buildUrl();
    
    const body = {
      messages,
      temperature: options.temperature ?? this.settings.temperature,
      max_tokens: options.maxOutputTokens ?? this.settings.maxOutputTokens,
      top_p: options.topP ?? this.settings.topP,
    };

    const response = await fetch(url, {
      method: "POST",
      headers: this.config.headers(),
      body: JSON.stringify(body),
      signal: options.abortSignal,
    });

    if (!response.ok) {
      throw new APICallError({
        statusCode: response.status,
        message: `Request failed with status ${response.status}`,
        url,
        requestBodyValues: body,
      });
    }

    const text = await response.text();
    const lines = text.split("\n").filter(Boolean);
    let fullContent = "";
    let stopReason: string | undefined;

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line) as BuilderSSEEvent;
        if (event.type === "delta" && event.delta) {
          fullContent += event.delta;
        }
        if (event.type === "done") {
          stopReason = event.stopReason;
        }
        if (event.type === "done" && event.actions?.[0]?.content) {
          fullContent = event.actions[0].content;
        }
      } catch {
        // Skip invalid JSON
      }
    }

    const content: LanguageModelV3Content[] = fullContent 
      ? [{ type: "text", text: fullContent }] 
      : [];

    const usage: LanguageModelV3Usage = {
      inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
      outputTokens: { total: undefined, text: undefined, reasoning: undefined },
    };

    return {
      content,
      finishReason: this.mapFinishReason(stopReason),
      usage,
      request: { body },
      response: { body: text },
      warnings: [],
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const messages = this.convertMessages(options.prompt);
    const url = this.buildUrl();

    const body = {
      messages,
      temperature: options.temperature ?? this.settings.temperature,
      max_tokens: options.maxOutputTokens ?? this.settings.maxOutputTokens,
      top_p: options.topP ?? this.settings.topP,
    };

    const response = await fetch(url, {
      method: "POST",
      headers: this.config.headers(),
      body: JSON.stringify(body),
      signal: options.abortSignal,
    });

    if (!response.ok) {
      throw new APICallError({
        statusCode: response.status,
        message: `Request failed with status ${response.status}`,
        url,
        requestBodyValues: body,
      });
    }

    const stream = response.body!
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(this.createSSEParser())
      .pipeThrough(this.createStreamTransformer());

    return { stream };
  }

  private buildUrl(): string {
    const baseURL = withoutTrailingSlash(this.config.baseURL);
    return `${baseURL}/codegen/completion?apiKey=${this.config.apiKey}&userId=${this.config.userId}`;
  }

  private convertMessages(prompt: LanguageModelV3CallOptions["prompt"]): BuilderCompletionMessage[] {
    return prompt.map((msg) => {
      const content = Array.isArray(msg.content) 
        ? msg.content.filter(p => p.type === "text").map(p => (p as any).text).join("")
        : typeof msg.content === "string" ? msg.content : "";
      
      if (msg.role === "system") {
        return { role: "system" as const, content };
      }
      if (msg.role === "user") {
        return { role: "user" as const, content };
      }
      return { role: "assistant" as const, content };
    });
  }

  private mapFinishReason(reason?: string): LanguageModelV3FinishReason {
    let unified: 'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other' = 'other';
    switch (reason) {
      case "end_turn":
        unified = "stop";
        break;
      case "max_tokens":
        unified = "length";
        break;
      case "stop_sequence":
        unified = "stop";
        break;
      case "tool_use":
        unified = "tool-calls";
        break;
    }
    return { unified, raw: reason };
  }

  private createSSEParser(): TransformStream<string, BuilderSSEEvent> {
    let buffer = "";

    return new TransformStream({
      transform(chunk, controller) {
        buffer += chunk;
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          
          try {
            const parsed = JSON.parse(trimmed);
            controller.enqueue(parsed);
          } catch {
            // Skip invalid JSON
          }
        }
      },

      flush(controller) {
        if (buffer.trim()) {
          try {
            const parsed = JSON.parse(buffer);
            controller.enqueue(parsed);
          } catch {
            // Skip invalid JSON
          }
        }
      },
    });
  }

  private createStreamTransformer(): TransformStream<BuilderSSEEvent, LanguageModelV3StreamPart> {
    let accumulatedText = "";
    let accumulatedReasoning = "";
    let hasStartedText = false;
    let finishReason: LanguageModelV3FinishReason = { unified: "other", raw: undefined };

    return new TransformStream({
      transform(chunk, controller) {
        // Emit text-start on first text delta
        if (!hasStartedText && (chunk.type === "delta" || chunk.type === "thinking" || chunk.type === "user")) {
          controller.enqueue({ type: "text-start", id: chunk.id || "0" });
          hasStartedText = true;
        }

        switch (chunk.type) {
          case "user":
            break;

          case "thinking":
            if (chunk.content) {
              accumulatedReasoning += chunk.content;
              controller.enqueue({
                type: "reasoning-delta",
                id: chunk.id || "0",
                delta: chunk.content,
              });
            }
            break;

          case "delta":
            if (chunk.content) {
              accumulatedText += chunk.content;
              controller.enqueue({ type: "text-delta", id: chunk.id || "0", delta: chunk.content });
            }
            break;

          case "done":
            finishReason = chunk.stopReason === "end_turn" 
              ? { unified: "stop", raw: chunk.stopReason }
              : chunk.stopReason === "max_tokens"
                ? { unified: "length", raw: chunk.stopReason }
                : { unified: "other", raw: chunk.stopReason };

            if (chunk.actions?.[0]?.content) {
              accumulatedText = chunk.actions[0].content;
            }
            break;
        }
      },

      flush(controller) {
        // Emit text-end if we had text
        if (hasStartedText || accumulatedText) {
          controller.enqueue({ type: "text-end", id: "0" });
        }

        controller.enqueue({
          type: "finish",
          finishReason,
          usage: {
            inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
            outputTokens: { total: accumulatedText.length, text: accumulatedText.length, reasoning: accumulatedReasoning.length || undefined },
          },
        });
      },
    });
  }
}

interface BuilderConfig {
  baseURL: string;
  apiKey: string;
  userId: string;
  headers: () => Record<string, string>;
}

export function createBuilder(options: BuilderProviderSettings): BuilderProvider {
  if (!options.apiKey) {
    throw new Error("Builder apiKey is required");
  }
  if (!options.userId) {
    throw new Error("Builder userId is required");
  }

  const baseURL = withoutTrailingSlash(options.baseURL) ?? "https://api.builder.io";
  const privateKey = options.privateKey ?? loadApiKey({
    apiKey: options.privateKey,
    environmentVariableName: "BUILDER_PRIVATE_KEY",
    description: "Builder.io private key",
  });

  const createModel = (modelId: string, settings: BuilderChatSettings = {}) =>
    new BuilderChatLanguageModel(modelId, settings, {
      baseURL,
      apiKey: options.apiKey,
      userId: options.userId,
      headers: () => ({
        Authorization: `Bearer ${privateKey}`,
        "Content-Type": "application/json",
      }),
    });

  const provider = function (modelId: string, settings?: BuilderChatSettings) {
    if (new.target) {
      throw new Error("The model factory function cannot be called with the new keyword.");
    }
    return createModel(modelId, settings);
  };

  provider.languageModel = createModel;

  return provider as BuilderProvider;
}
