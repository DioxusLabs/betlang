module Main where

import Data.List (intercalate, sort)
import qualified Data.Map.Strict as Map

counts :: [String] -> Map.Map String Int
counts = foldr (\name acc -> Map.insertWith (+) name 1 acc) Map.empty

render :: Map.Map String Int -> String
render table =
  intercalate "\n" [key <> "=" <> show value | (key, value) <- Map.toList table]

main :: IO ()
main = do
  let names = sort ["rust", "python", "rust"]
  putStrLn (render (counts names))
