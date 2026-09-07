# Testing preference

For subsequent dataset/pipeline tests, do not visually compare input images with
captions, regions, masks, or other tool output. The user explicitly requested
avoiding token expenditure on this review. Do not render contact sheets or open
images for this purpose unless the user explicitly asks for visual review.

Verify technical behavior instead: completion, errors, file formats and schemas,
hashes, image/mask dimensions, coordinate bounds, timing, and memory metrics.
Report technical success without claiming semantic or visual quality validation.

# Interface language

All application UI text must be in English, including labels, dialogs, tooltips,
status messages, errors, and accessibility names. The conversation language does
not determine the application language. User-provided captions and filenames
retain their original language.
