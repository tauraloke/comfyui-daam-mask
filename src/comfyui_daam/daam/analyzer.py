from itertools import chain
from comfy.text_encoders.flux import FluxClipModel


class PromptAnalyzer:
    def __init__(self, clip, all_tokens, model=None):
        self.clip = clip
        self.tokens = all_tokens

        if model is None:
            if isinstance(clip.cond_stage_model, FluxClipModel):
                self.model = "t5xxl"
            else:
                # SDXL & SD1.5
                self.model = "l"
        else:
            self.model = model

        self.end_token = self._get_end_token()

    def _get_end_token(self):
        if self.model == "l":
            return self.clip.cond_stage_model.clip_l.special_tokens["end"]
        elif self.model == "t5xxl":
            return 1  # </s>

    def _get_tokens_list(self, tokens):
        return list(chain.from_iterable(tokens[self.model]))

    def encode(self, text: str):
        tokens = self.clip.tokenize(text)
        tokens = self._get_tokens_list(tokens)

        if self.model == "t5xxl":
            start_idx = 0
        else:
            start_idx = 1

        for i in range(len(tokens)):
            if tokens[i][0] == self.end_token:
                return tokens[start_idx:i]

    def calc_word_indecies(self, word: str, limit: int = -1, start_pos=0):
        word = word.lower()
        merge_idxs = []

        tokens = self.tokens
        tokens = self._get_tokens_list(tokens)
        needles = self.encode(word)

        limit_count = 0
        current_pos = 0

        if len(needles) == 0:
            return merge_idxs, start_pos

        for i, token in enumerate(tokens):
            current_pos = i
            if i < start_pos:
                continue

            if needles[0][0] == token[0] and len(needles) > 1:
                next = i + 1
                success = True
                for needle in needles[1:]:
                    if next >= len(tokens) or needle[0] != tokens[next][0]:
                        success = False
                        break
                    next += 1

                # append consecutive indexes if all pass
                if success:
                    merge_idxs.extend(list(range(i, next)))
                    if limit > 0:
                        limit_count += 1
                        if limit_count >= limit:
                            break

            elif needles[0][0] == token[0]:
                merge_idxs.append(i)
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        return merge_idxs, current_pos
