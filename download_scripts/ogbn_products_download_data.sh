

if [ ! -d "gpt_responses/ogbn-products" ]; then
    # ogbn_arxiv_orig GPT responses
gdown 1C769tlhd8pT0s7I3vXIEUI-PK7A4BB1p
unzip ogbn-products.zip
mkdir -p gpt_responses
mv ogbn-products gpt_responses/
rm -rf ogbn-products.zip
fi

# parse the responses to get GPT preds
python core/data_utils/parse_ogbn_gpt_preds.py
