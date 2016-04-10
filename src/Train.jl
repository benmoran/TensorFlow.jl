module Train
using ..CoreTypes
using ..API

export minimize

function minimize{T<:AbstractTensor}(opt::Optimizer,
                                     cost::T,
                                     variables::Vector{Variable}=Variable[])
  Operation(opt.x[:minimize](cost.x, var_list=(isempty(variables) ? nothing : [v.x for v in variables])))
end

end
