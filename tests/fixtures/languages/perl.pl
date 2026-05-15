use strict;
use warnings;
use feature 'say';

sub count_languages {
    my (@values) = @_;
    my %counts;
    for my $value (@values) {
        $value =~ s/^\s+|\s+$//g;
        $counts{lc $value}++;
    }
    return \%counts;
}

my $counts = count_languages(qw(Rust Rust Python));
for my $language (sort keys %{$counts}) {
    say "$language=$counts->{$language}";
}
