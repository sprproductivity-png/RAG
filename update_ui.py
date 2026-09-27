import re

with open('frontend/src/components/SearchStudio.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Headings
content = content.replace('Search Studio', 'Ask Studio (NotebookLM Style)')
content = content.replace('Hybrid Retrieval & Re-ranking', 'Generative QA directly backed by your documents')

# 2. Force Enter to trigger handleAsk
content = content.replace("onKeyDown={(e) => e.key === 'Enter' && handleSearch()}", "onKeyDown={(e) => e.key === 'Enter' && handleAsk()}")

# 3. Force Quick-fill chips to trigger handleAsk
content = content.replace('handleSearch(q);', 'handleAsk(q);')

# 4. Hide/Remove the "Search" button right next to the input
search_button_regex = r'<button[^>]*onClick=\{\(\) => handleSearch\(\)\}[^>]*>.*?<span>Search</span>\s*</button>'
content = re.sub(search_button_regex, '', content, flags=re.DOTALL)

# 5. Fix "Ready to search" placeholder
content = content.replace('Ready to search', 'Ready to Ask')
content = content.replace('Execute Dense and BM25 sparse retrieval fused with Cross-Encoder re-ranking.', 'Ask questions and get AI-generated answers based strictly on your source documents.')

# 6. Change "Search Results" to "Sources"
content = content.replace('{/* Search Results */}', '{/* AI Answer & Sources */}')

with open('frontend/src/components/SearchStudio.jsx', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated SearchStudio successfully.")
