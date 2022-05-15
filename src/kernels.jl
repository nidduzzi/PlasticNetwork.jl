using Random, Distributions, GPUArrays, CUDA, NNlib
function neuronKernel(network::Network, out::Array, ∇w, ∇b, ∑w::Array, ∑b::Array, srcIdx, notSrcIdx, inputIdx, outputIdx, χ, inputNum::Int, samplingRate::Int, outputLength::Int, ϵ::Float64)
    index = threadIdx().x;
    stride = blockDim().x;
end