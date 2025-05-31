import re


class TextCleaner:

    def __init__(self):
        self.termination_keywords = {
            "See also", "References", "Notes", "External links", "Sources", "Primary sources",
            "Further reading", "Interviews", "Works cited", "Official website", "Notes and references"
        }

    def clean_text(self, text):

        if not text or not text.strip():
            return ""

        lines = text.strip().split('\n')
        page_title = lines[0].strip() if lines else ""
        remaining_lines = lines[1:] if len(lines) > 1 else []

        content_lines = self._remove_sections_after_keywords(remaining_lines)

        filtered_lines = self._remove_short_lines(content_lines)

        cleaned_text = self._basic_normalization('\n'.join(filtered_lines))

        return page_title, cleaned_text.strip()

    def _remove_sections_after_keywords(self, lines):

        content_lines = []

        for line in lines:
            line_stripped = line.strip()

            should_terminate = False
            for keyword in self.termination_keywords:
                if line_stripped == keyword or line_stripped.startswith(keyword):
                    should_terminate = True
                    break

            if should_terminate:
                break

            content_lines.append(line)

        return content_lines

    def _remove_short_lines(self, lines):

        filtered_lines = []

        for line in lines:
            line_stripped = line.strip()

            if not line_stripped:
                continue

            word_count = len(line_stripped.split())

            if word_count > 4:
                filtered_lines.append(line_stripped)

        return filtered_lines

    def _basic_normalization(self, text):

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove empty lines and excessive line breaks
        text = re.sub(r'\n\s*\n', '\n', text)

        # Basic punctuation spacing fixes
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        return text

    def get_page_title(self, text):

        lines = text.strip().split('\n')
        return lines[0].strip() if lines else ""
