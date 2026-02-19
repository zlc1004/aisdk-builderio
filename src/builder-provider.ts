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
import { loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils";
import os from "node:os";

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
}

export interface BuilderProvider {
  (modelId: string, settings?: BuilderChatSettings): BuilderChatLanguageModel;
  languageModel(modelId: string, settings?: BuilderChatSettings): BuilderChatLanguageModel;
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
    const { messages, userPrompt } = this.convertMessages(options.prompt);
    const url = this.buildUrl();
    
    // Build CodeGenInputOptions payload
    const body: any = {
      position: "cli", // Default position
      sessionId: (options as any).sessionId || "session-" + Date.now(),
      userPrompt,
      codeGenMode: "quality-v4",
      userContext: await this.getUserContext(),
      maxTokens: options.maxOutputTokens ?? this.settings.maxOutputTokens,
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
    let stopReason: LanguageModelV3FinishReason = { unified: "other", raw: undefined };

    for (const line of lines) {
      try {
        const event = JSON.parse(line);
        if (event.type === "delta" && event.content) {
          fullContent += event.content;
        } else if (event.type === "text" && event.content) {
          fullContent += event.content;
        } else if (event.type === "done") {
          stopReason = this.mapFinishReason(event.stopReason);
        }
      } catch {
        // Skip invalid JSON
      }
    }

    const content: LanguageModelV3Content[] = [{ type: "text", text: fullContent }];

    return {
      content,
      finishReason: stopReason.unified as any,
      usage: {
        inputTokens: 0,
        outputTokens: fullContent.length,
      } as any,
      request: { body },
      response: { body: text },
      warnings: [],
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const { userPrompt } = this.convertMessages(options.prompt);
    const url = this.buildUrl();

    const body: any = {
      position: "cli",
      sessionId: (options as any).sessionId || "session-" + Date.now(),
      userPrompt,
      codeGenMode: "quality-v4",
      userContext: await this.getUserContext(),
      maxTokens: options.maxOutputTokens ?? this.settings.maxOutputTokens,
      // Map tools if provided
      enabledTools: options.tools?.map(t => t.name),
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

  private convertMessages(prompt: LanguageModelV3CallOptions["prompt"]): { messages: any[], userPrompt: string } {
    let userPrompt = "";
    const messages = prompt.map((msg) => {
      const content = Array.isArray(msg.content) 
        ? msg.content.filter(p => p.type === "text").map(p => (p as any).text).join("")
        : typeof msg.content === "string" ? msg.content : "";
      
      if (msg.role === "user") {
        userPrompt = content; // Last user prompt wins for Builder
      }
      return { role: msg.role, content };
    });

    return { messages, userPrompt };
  }

  private async getUserContext() {
    return {
      client: "@builder.io/ai-sdk-provider",
      clientVersion: "1.0.0",
      nodeVersion: process.version,
      systemPlatform: process.platform,
      systemEOL: os.EOL,
      systemArch: os.arch(),
    };
  }

  private mapFinishReason(reason?: string): LanguageModelV3FinishReason {
    switch (reason) {
      case "end_turn":
        return { unified: "stop", raw: reason };
      case "max_tokens":
        return { unified: "length", raw: reason };
      case "tool_use":
        return { unified: "tool-calls", raw: reason };
      case "content_filter":
        return { unified: "content-filter", raw: reason };
      default:
        return { unified: "other", raw: reason };
    }
  }

  private createSSEParser(): TransformStream<string, any> {
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
            controller.enqueue(JSON.parse(trimmed));
          } catch { /* ignore */ }
        }
      },
      flush(controller) {
        if (buffer.trim()) {
          try {
            controller.enqueue(JSON.parse(buffer));
          } catch { /* ignore */ }
        }
      },
    });
  }

  private createStreamTransformer(): TransformStream<any, LanguageModelV3StreamPart> {
    let hasStartedText = false;
    let accumulatedText = "";
    let accumulatedReasoning = "";
    const self = this;

    return new TransformStream({
      transform(chunk, controller) {
        if (!hasStartedText && (chunk.type === "delta" || chunk.type === "thinking" || chunk.type === "text")) {
          controller.enqueue({ type: "text-start", id: chunk.id || "0" });
          hasStartedText = true;
        }

        switch (chunk.type) {
          case "thinking":
            if (chunk.content) {
              accumulatedReasoning += chunk.content;
              controller.enqueue({ type: "reasoning-delta", id: chunk.id || "0", delta: chunk.content });
            }
            break;
          case "delta":
          case "text":
            if (chunk.content) {
              accumulatedText += chunk.content;
              controller.enqueue({ type: "text-delta", id: chunk.id || "0", delta: chunk.content });
            }
            break;
          case "tool":
            // Map Builder tool to AI SDK tool-call
            controller.enqueue({
              type: "tool-call",
              toolCallId: chunk.id || "tool-" + Date.now(),
              toolName: chunk.name,
              args: chunk.content ? JSON.parse(chunk.content) : {},
            } as any);
            break;
          case "error":
            if (chunk.code === "ask-to-continue") {
              // Custom handling for ask-to-continue
              // For now, we emit as a specific error or message
              controller.enqueue({
                type: "text-delta",
                id: chunk.id || "0",
                delta: `\n[Wait: ${chunk.message}]\n`,
              });
            }
            break;
          case "done":
            controller.enqueue({
              type: "finish",
              finishReason: self.mapFinishReason(chunk.stopReason).unified as any,
              usage: {
                inputTokens: 0,
                outputTokens: accumulatedText.length,
              } as any,
            });
            break;
        }
      },
      flush(controller) {
        if (hasStartedText) {
          controller.enqueue({ type: "text-end", id: "0" });
        }
      }
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
    return createModel(modelId, settings);
  };

  provider.languageModel = createModel;

  return provider as BuilderProvider;
}
