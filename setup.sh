mkdir -p ~/.streamlit/
echo "\
[servers]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
