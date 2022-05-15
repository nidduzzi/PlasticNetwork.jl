module PlasticNetwork
export Network, propagate!, network_copy!
using Random, Distributions, GPUArrays, CUDA, NNlib, ProgressMeter
mutable struct Network{T <: AbstractArray}
    w::T # Synapse weight
    δw::T # weight learning rate (relation to w is transposed)
    α::T # Synapse alive or not
    accumulator::T
	cycle_timer::T
    b::T # Neuron bias
    δb::T # bias learning rate
    inputNeurons::T
    outputNeurons::T
    signalSrc::T
    signal::T
    size::Int
    connectivity::AbstractFloat
    selfConnectivity::AbstractFloat
    T::Type
    function Network{T}(size::Int = 10, connectivity::AbstractFloat = 0.1, selfConnectivity::AbstractFloat = 0.1) where {T <: AbstractArray}
        # synapse weights
        w::T = Float32.(rand(truncated(Normal(0.0, connectivity), -1, 1), (size, size)))
        δw::T = Float32.(rand(truncated(Normal(0, 0.001 * connectivity), 0, 1), (size, size)))
        α::T = Bool.(rand(Binomial(1, connectivity), (size, size)))
        # neuron biases
        accumulator::T = zeros(Int, size)
        cycle_timer::T = zeros(Int, size)
        b::T = Float32.(rand(truncated(Normal(0.0, connectivity), -1, 1), (size,)))
        δb::T = Float32.(rand(truncated(Normal(0.0, 0.001 * connectivity), 0, 1), (size,)))
        if selfConnectivity != connectivity
            w[T([CartesianIndex(i, i) for i in 1:size])] .*= T(BitArray(rand(Binomial(1, selfConnectivity / connectivity), size)))
        end
        inputNeurons::T = falses(size)
        outputNeurons::T = falses(size)
        signal::T = zeros(Float16, size)
        signalSrc::T = falses(size)
        new{T}(w, δw, α, accumulator, cycle_timer, b, δb, inputNeurons, outputNeurons, signalSrc, signal, size, connectivity, selfConnectivity, T)
    end
    
    function Network{T}(network::Network) where {T <: AbstractArray}
        w = deepcopy(network.w)
        δw = deepcopy(network.δw)
        α = deepcopy(network.α)
        accumulator = network.accumulator
        cycle_timer = network.cycle_timer
        b = deepcopy(network.b)
        δb = deepcopy(network.δb)
        inputNeurons = deepcopy(network.inputNeurons)
        outputNeurons = deepcopy(network.outputNeurons)
        signal = deepcopy(network.signal)
        signalSrc = deepcopy(network.signalSrc)
        connectivity = network.connectivity
        selfConnectivity = network.selfConnectivity
        new{T}(w, δw, α, accumulator, cycle_timer, b, δb, inputNeurons, outputNeurons, signalSrc, signal, network.size, network.connectivity, network.selfConnectivity, network.T)
    end
end

function network_copy!(dest::Network, src::Network)::Nothing
    dest.w = dest.T(deepcopy(src.w))
    dest.δw = dest.T(deepcopy(src.δw))
    dest.α = dest.T(deepcopy(src.α))
    dest.accumulator = dest.T(deepcopy(src.accumulator))
    dest.cycle_timer = dest.T(deepcopy(src.cycle_timer))
    dest.b = dest.T(deepcopy(src.b))
    dest.δb = dest.T(deepcopy(src.δb))
    dest.inputNeurons = dest.T(deepcopy(src.inputNeurons))
    dest.outputNeurons = dest.T(deepcopy(src.outputNeurons))
    dest.signal = dest.T(deepcopy(src.signal))
    dest.signalSrc = dest.T(deepcopy(src.signalSrc))
    dest.connectivity = src.connectivity
    dest.selfConnectivity = src.selfConnectivity
    dest.size = src.size
    return nothing
end

function propagate_loop!(network::Network, out::Array, ∇w, ∇b, ∑w::Array, ∑b::Array, srcIdx, notSrcIdx, inputIdx, outputIdx, χ, inputNum::Int, samplingRate::Int, outputLength::Int, ϵ::Float64)
    @showprogress 1 for x in 1:inputNum
        # initialize weight change matrix and bias change vector for each input with zeros
        # TODO: implement pipelining of inputs and immediate updates of weights
        # TODO: implement combination/merger/integration of inbound signal (next input) with current signals that pass through the input neurons
        
        # * modify signal elements that pass through input neurons with the inbound signals
        network.signal[inputIdx] .= sigmoid.(network.signal[inputIdx] .+ χ[:,x])
        
        # * pool the updates of an iteration to the weight & bias change arrays
        @inbounds ∇w[srcIdx,:] .= (∇w - (network.δw .* network.signal .* map(x -> sign(x), network.w')))[srcIdx,:]
        @inbounds ∇b[srcIdx] .= (∇b - (network.δb .* network.signal .* map(x -> sign(x), network.b)))[srcIdx]
        
        # TODO: compute the probability of a synapse becoming viable or nonviable
        # TODO:update active synapses matrix α
        #= 
        # * This can be implemented with the adaptive synaptogenesis method
        # * or the Invariant information clustering on the neuron level =#
        # TODO: pass signal through network from signalSrc
        @inbounds network.signal = ((network.α .* network.w) * (network.signalSrc .* network.signal))
        # * update synapse weight between signal source and signal receiving neuron
        let αidx::network.T = ((network.T == CuArray) ? CUDA.findall(network.α .* (network.signalSrc')) : findall(network.α .* (network.signalSrc')))
            @inbounds network.w[αidx] .= tanh.((network.w + (∇w .* ϵ)')[αidx])
        end
        # * update signalSrcNeurons to receiving neurons and exclude signals that are approximately 0
        @inbounds network.signalSrc .= (.!isapprox.(network.signal, 0.0)) .| network.inputNeurons
        srcIdx = findall(network.signalSrc)
        notSrcIdx = findall(el -> !el, network.signalSrc)
        # * increment accumulator and restart timer for neurons that fire
        # TODO: make two branches of the loop, one for cpu and the other for gpu
        @inbounds network.accumulator[srcIdx] .+= 1
        @inbounds network.cycle_timer[srcIdx] .= 0
        # * and decrement accumulator and start/continue timer for neurons that don't fire
        @inbounds network.accumulator[notSrcIdx] .-= 1
        @inbounds network.cycle_timer[notSrcIdx] .+= 1
        # * update signal
        @inbounds network.signal .= sigmoid.(network.signal + (network.b .* network.signalSrc))
        # * update biases of receiving neurons
        @inbounds network.b[srcIdx] .= tanh.((network.b + (∇b .* ϵ))[srcIdx])
        
        #= 
        # TODO: The network needs to be told why the output neurons are important via feedback
        # * this can be done by specifying a structure in the network explicitly that either calculates the error, or mutual information
        # * or it can be done by processing and modifying the signal passing through the output neurons
        # ** signal modification for UNSUPERVISED learning can be done by things like mutual information in Invariant information clustering (Ji, X., Henriques, J.F. & Vedaldi, A., 2019. Invariant Information Clustering for Unsupervised Image Classification and Segmentation. arXiv:1807.06653 [cs]. Available at: http://arxiv.org/abs/1807.06653.)
        # ** signal modifcation for SUPERVISED learning can be done by replacing the orginal signal with the error =#
        # * extract outbound signals from output neurons
        if x % samplingRate == 0
            @inbounds local outboundSignalSrc::BitArray{1} = Array(network.signalSrc .& network.outputNeurons)
            if reduce(|, outboundSignalSrc)
                @inbounds local _out::Array{Float16,1} = zeros(Float16, outputLength)
                @inbounds _out[outboundSignalSrc[Array(outputIdx)]] = Array(network.signal)[findall(outboundSignalSrc)]
                @inbounds push!(out, _out)
            end
            # track weight changes
            @inbounds push!(∑w, Array(Float16.(∇w)))
            @inbounds push!(∑b, Array(Float16.(∇b)))
        end
    end
end

function propagate!(network::Network, χ, outputLength::Integer; ϵ::Float64 = 0.01, samplingRate::Int = 10)
	# Initialize propagation variables
    local out = Array{Array{Float16,1},1}()
    local ∑w = Array{Array{Float16,2},1}()
    local ∑b = Array{Array{Float16,1},1}()
    if !isa(χ, network.T)
        error("mismatch input and network strorage types")
    else
        local inputLength = Base.length(χ[:,1])
        local inputNum = size(χ)[2]
        if sum(network.inputNeurons) < inputLength
            # ! bad setting of input neurons
            # TODO: FIX THIS ISSUE
            println("current network input length is smaller than input length, adding $(inputLength - sum(network.inputNeurons))")
            @inbounds network.inputNeurons[sample(Array(findall(el -> !el, network.inputNeurons .& (.!network.outputNeurons))), inputLength - sum(network.inputNeurons), replace = false)] .= true
            @inbounds network.signalSrc = network.inputNeurons
        elseif sum(network.inputNeurons) > inputLength
            println("current network input length is larger than input length, removing $(sum(network.inputNeurons) - inputLength)")
            @inbounds network.inputNeurons[sample(Array(findall(network.inputNeurons)), sum(network.inputNeurons) - inputLength, replace = false)] .= false
            @inbounds network.signalSrc = network.inputNeurons
        end
        if sum(network.outputNeurons) < outputLength
            println("current network output length is smaller than output length, adding $(outputLength - sum(network.outputNeurons))")
            @inbounds network.outputNeurons[sample(Array(findall(el -> !el, network.outputNeurons .& (.!network.inputNeurons))), outputLength - sum(network.outputNeurons), replace = false)] .= true
        elseif sum(network.outputNeurons) > outputLength
            println("current network output length is larger than output length, removing $(sum(network.outputNeurons) - outputLength)")
            @inbounds network.outputNeurons[sample(Array(findall(network.outputNeurons)), sum(network.outputNeurons) - outputLength, replace = false)] .= false
        end
        
        local it = 1
        local inboundSignalSrc::network.T = falses(network.size)
        local ∇w::network.T = zeros(Float32, (network.size, network.size))
        local ∇b::network.T = zeros(Float32, (network.size))
        print(string("number of input: $inputNum\ninput_length: $inputLength\noutput_size: $outputLength\n"))
        local srcIdx::network.T = findall(network.signalSrc)
        local notSrcIdx::network.T = findall(el -> !el, network.signalSrc)
        local inputIdx::network.T = findall(network.inputNeurons)
        local outputIdx::network.T = findall(network.outputNeurons)
        propagate_loop!(network, out, ∇w, ∇b, ∑w, ∑b, srcIdx, notSrcIdx, inputIdx, outputIdx, χ, inputNum, samplingRate, outputLength, ϵ)
        # TODO: propagate until the last input reaches the output
        # while false
            
        # end
    end
	return size(network.signal), out, ∑w, ∑b
end



end
