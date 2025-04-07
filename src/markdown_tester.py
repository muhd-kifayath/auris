from markitdown import MarkItDown
import re

class DocumentMarkdown:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_content = ""

    def convert(self):
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result
    
    def chunk_text(self, text):
        """Group each heading and its following content into a single chunk."""

        # # This pattern captures each heading and its content until the next heading
        # pattern = re.compile(r'(#{1,6} .+?)(?=\n#{1,6} |\Z)', re.DOTALL)
        # matches = pattern.findall(text)

        # if matches:
        #     chunks = [match.strip() for match in matches if match.strip()]
        # else:
            # Fallback to splitting by paragraphs if no headings found
        chunks = [para.strip() for para in text.split("\n\n") if para.strip()]

        return chunks


    def process_document(self):
        """Processes a document (PDF/Word), extracts text, and replaces images with extracted text.
        Splits text into chunks for better vector storage and retrieval.
        """
        # Convert the document to markdown
        self.text_content = self.convert()

        print(f"Text content: {self.text_content}")
        
        # Chunk the text
        chunks = self.chunk_text(self.text_content.text_content)
        
        return chunks

cd = DocumentMarkdown("documents/file.pdf")
chunks = cd.process_document()
for i in range(len(chunks)):
    print(f"Chunked Text {i}: {chunks[i]}\n")


