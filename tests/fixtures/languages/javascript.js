const names = ["Ada", "Grace", "Linus"];

function greet(name) {
  return `hello ${name}`;
}

const messages = names
  .filter((name) => name.length > 0)
  .map((name) => greet(name));

for (const message of messages) {
  console.log(message);
}
