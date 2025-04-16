class PromptAnalyzer:
    def __init__(self, clip, all_tokens):
        self.clip = clip
        self.tokens = all_tokens

    def encode(self, text: str):
        tokens = self.clip.tokenize(text)['g'][0]
        for i in range(len(tokens)):
            if tokens[i][0] == self.clip.cond_stage_model.clip_g.special_tokens['end']:
                return tokens[1:i]

    def calc_word_indecies(self, word: str, limit: int = -1, start_pos=0):
        word = word.lower()
        merge_idxs = []

        tokens = self.tokens['g'][0]
        needles = self.encode(word)

        limit_count = 0
        current_pos = 0
        for i, token in enumerate(tokens):
            current_pos = i
            if i < start_pos:
                continue

            if needles[0] == token and len(needles) > 1:
                next = i + 1
                success = True
                for needle in needles[1:]:
                    if next >= len(tokens) or needle != tokens[next]:
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

            elif needles[0] == token:
                merge_idxs.append(i)
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        return merge_idxs, current_pos
