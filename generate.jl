using Pkg
Pkg.activate(".")
using JSON
using WordTokenizers
using StatsBase
using Flux,CuArrays
using Flux:onehot
using Base.Iterators:partition
using Metalhead
using JLD
using BSON:@save,@load

function sample(image_path)
    img = Metalhead.preprocess(load(image_path)) |> gpu
    features = vgg(img) |> cpu
    
    reset()
    prev_word = "<s>"
    lstm_inp = cnn_encoder(features)
    lstm_out = rnn_decoder(lstm_inp)
    output = ""
    
    for i in 1:15
        output = string(output," ",prev_word)
        if prev_word == "</s>"
            break
        end
        word_embeddings = embedding(onehotword(prev_word))
        predictions = softmax(decoder(rnn_decoder(word_embeddings)))
        next_word = idx2word[Flux.argmax(predictions)[1]]
        prev_word = next_word
    end
    
    output
end

vgg = VGG19() |> gpu
Flux.testmode!(vgg)
vgg = vgg.layers[1:end-3] |> gpu

punc = "!#%&()*+.,-/:;=?@[]^_`{|}~"
punctuation = [punc[i] for i in 1:length(punc)]
data = load_data(BASE_PATH,10,punctuation)
image_names = [d[2] for d in data]

println(sample(image_names[1]))
