import re
html=open("web/static/index.html").read()
start=html.index("<script>")+8
end=html.index("</script>")
js=html[start:end]
print(f"JS length: {len(js)} chars")
print(f"Braces: open={js.count(chr(123))} close={js.count(chr(125))} diff={js.count(chr(123))-js.count(chr(125))}")
print(f"Parens: open={js.count(chr(40))} close={js.count(chr(41))} diff={js.count(chr(40))-js.count(chr(41))}")
print(f"Brackets: open={js.count(chr(91))} close={js.count(chr(93))} diff={js.count(chr(91))-js.count(chr(93))}")
