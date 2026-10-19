# Based on seminar materials

# Don't forget to support cases when target_text == ''  🫡🫡🫡🫡

#       /\__/\
#      /`    `\
#    === 0  0 ===
#      \  --  /
#      /        \
#     /          \
#
#    |            |
#     \  ||  ||  /
#      \_||__||_/

# наконец-то алгоритмы пригодились


def calc_cer(target_text, predicted_text) -> float:
    # TODO
    dp = [list(range(len(predicted_text) + 1)), [0] * (len(predicted_text) + 1)]
    for i, target_char in enumerate(target_text, 1):
        dp[1][0] = i
        for j, predicted_char in enumerate(predicted_text, 1):
            dp[1][j] = min(
                dp[1][j - 1] + 1,
                dp[0][j] + 1,
                dp[0][j - 1] + (predicted_char != target_char),
            )
        dp[0] = dp[-1].copy()
    return dp[0][-1], len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    # TODO
    target_text = target_text.split()
    predicted_text = predicted_text.split()
    dp = [list(range(len(predicted_text) + 1)), [0] * (len(predicted_text) + 1)]
    for i, target_char in enumerate(target_text, 1):
        dp[1][0] = i
        for j, predicted_char in enumerate(predicted_text, 1):
            dp[1][j] = min(
                dp[1][j - 1] + 1,
                dp[0][j] + 1,
                dp[0][j - 1] + (predicted_char != target_char),
            )
        dp[0] = dp[-1].copy()
    return dp[0][-1], len(target_text)
