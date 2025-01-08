import { AutoTokenizer } from '@huggingface/transformers';
import ort from 'onnxruntime-node';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function softmax(logits) {
    // Convert input to array if needed
    const arr = Array.isArray(logits) ? logits : Array.from(logits);
    // Find max for numerical stability
    const maxLogit = Math.max(...arr);
    // Compute exp(logits - max)
    const expLogits = arr.map(x => Math.exp(x - maxLogit));
    // Compute sum for normalization
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    // Normalize to get probabilities
    return expLogits.map(x => x / sumExp);
}

const chatExample = [
    { role: "user", content: "What's the weather like today?" },
    { role: "assistant", content: "It's sunny and warm." },
    { role: "user", content: "I like the weather. but" },
    { role: "user", content: "I'm not sure what to do." }
];

async function initializeModel() {
    try {
        // download the model save to onnx folder
        const localPathOnnx = path.resolve(__dirname, 'onnx/model_quantized.onnx');
        const session = await ort.InferenceSession.create(localPathOnnx);
        const tokenizer = await AutoTokenizer.from_pretrained('livekit/turn-detector');
        const eou_index = tokenizer.encode("<|im_end|>")[0];
        console.log('End of turn token:', eou_index);
        
        return { tokenizer, session, eou_index };
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}
const PUNCS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'; // Punctuation without single quote

function normalize(text) {
    // Remove punctuation and normalize whitespace
    const stripped = text.replace(new RegExp(`[${PUNCS}]`, 'g'), '');
    return stripped.toLowerCase().trim().split(/\s+/).join(' ');
}

async function formatChatContext(chatContext, tokenizer) {
    const normalizedContext = chatContext
        .map(msg => {
            const content = normalize(msg.content);
            return content ? { ...msg, content } : null;
        })
        .filter(Boolean);

    const convoText = tokenizer.apply_chat_template(normalizedContext, {
        add_generation_prompt: false,
        add_special_tokens: false,
        tokenize: false,
        add_generation_prompt: true,
      });
    console.log('ConvoText:', convoText);
    const eouToken = "<|im_end|>";
    const lastEOUIndex = convoText.lastIndexOf(eouToken);
    return lastEOUIndex >= 0 ? convoText.slice(0, lastEOUIndex) : convoText;
}

async function predictEndOfTurn(chatContext, { tokenizer, session, eou_index }) {
    const formattedText = await formatChatContext(chatContext, tokenizer);
    
    const inputs = await tokenizer(formattedText, {
        return_tensors: 'np',
        dtype: 'int64'
    });
    
    const output = await session.run({
        input_ids: inputs.input_ids,
        dtype : 'int64'
    });
    const data = output.logits.data;
    const dims = output.logits.dims;
    const lastTokenLogits = data.slice(-dims[2]);
    const probs = softmax(lastTokenLogits);
    const probability = probs[eou_index];
    return probability;
}

initializeModel().then(async (model) => {
    // count the time
    console.time('predictEndOfTurn');
    const probability = await predictEndOfTurn(chatExample, model);
    console.log('End of turn probability:', probability);
    console.timeEnd('predictEndOfTurn');
}).catch(console.error);