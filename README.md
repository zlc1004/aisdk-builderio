# ai-sdk-provider-builder

custom ai sdk provider for builder.io

made by kobosh

## usage

```ts
import { createBuilder } from 'ai-sdk-provider-builder';
import { generateText } from 'ai';

async function main() {
  const builder = createBuilder({
    // found in your node_modules/.builder/data.json
    apiKey: 'apiKey', // user api key
    userId: 'userId', // credentials.userId
    privateKey: 'bpk-privateKey', // credentials.builderPrivateKey
  });

  const { text } = await generateText({
    model: builder('builder-model'), // modelId is required but currently unused by our internal mapping
    prompt: 'what is your name?',
  });
  console.log(text);
}

main();

```
