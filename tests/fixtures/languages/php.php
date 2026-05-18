<?php

declare(strict_types=1);

final class Greeter
{
    public function __construct(private string $name)
    {
    }

    public function message(): string
    {
        return "hello {$this->name}";
    }
}

$names = ['Ada', 'Grace', 'Linus'];
foreach ($names as $name) {
    echo (new Greeter($name))->message(), PHP_EOL;
}
