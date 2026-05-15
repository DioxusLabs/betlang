local M = {}

function M.greet(name)
  return string.format("hello %s", name)
end

function M.count(values)
  local counts = {}
  for _, value in ipairs(values) do
    counts[value] = (counts[value] or 0) + 1
  end
  return counts
end

for key, value in pairs(M.count({ "lua", "lua", "rust" })) do
  print(key, value)
end

return M
