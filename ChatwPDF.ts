// app/api/chat/route.ts
import { streamText, embed } from 'ai';
import { google } from '@ai-sdk/google';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

// 1. Define our types for type safety
interface VectorSearchResult {
  text: string;
  similarity: number;
}

// 2. Mock Vector Database Search Function
// In a real app, this connects to Pinecone, Supabase pgvector, Qdrant, etc.
async function searchVectorDatabase(embedding: number[]): Promise<VectorSearchResult[]> {
  console.log(`Searching DB with a ${embedding.length}-dimensional vector...`);
  
  // Simulating the chunks of text you would retrieve from a parsed PDF
  return [
    { text: "The company's Q3 revenue grew by 15% year-over-year to $45M.", similarity: 0.92 },
    { text: "Operating expenses decreased by $2M due to cloud infrastructure optimizations.", similarity: 0.85 }
  ];
}

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    // 3. Grab the user's latest question from the message array
    const latestMessage = messages[messages.length - 1];

    if (!latestMessage || latestMessage.role !== 'user') {
      return new Response("Invalid message format", { status: 400 });
    }

    // 4. Generate an embedding for the user's search query
    // This turns the text into numbers so we can do similarity math
    const { embedding } = await embed({
      model: google.textEmbeddingModel('text-embedding-004'),
      value: latestMessage.content,
    });

    // 5. Search the Vector DB for chunks of the PDF relevant to the question
    const relevantChunks = await searchVectorDatabase(embedding);
    
    // Combine the retrieved chunks into one large text block
    const contextString = relevantChunks.map(chunk => chunk.text).join('\n---\n');

    // 6. Construct a dynamic system prompt
    // We strictly instruct the AI to ONLY use the injected context.
    const systemPrompt = `
      You are an expert document analyst. Answer the user's question based strictly on the context provided below.
      If the context does not contain the answer, reply exactly with: "I do not have enough information in the document to answer that."
      
      <context>
      ${contextString}
      </context>
    `;

    // 7. Call the AI and stream the response
    const result = await streamText({
      model: google('gemini-1.5-pro-latest'),
      messages: messages,
      system: systemPrompt,
      temperature: 0.2, // Lower temperature makes the AI less creative and more factual
    });

    return result.toDataStreamResponse();

  } catch (error) {
    console.error("Error in RAG pipeline:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}