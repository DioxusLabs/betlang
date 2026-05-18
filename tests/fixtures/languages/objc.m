#import <Foundation/Foundation.h>

@interface BLGreeter : NSObject
@property(nonatomic, copy) NSString *name;
- (instancetype)initWithName:(NSString *)name;
- (NSString *)message;
@end

@implementation BLGreeter
- (instancetype)initWithName:(NSString *)name {
    self = [super init];
    if (self) {
        _name = [name copy];
    }
    return self;
}

- (NSString *)message {
    return [NSString stringWithFormat:@"hello %@", self.name];
}
@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSArray<NSString *> *names = @[@"Ada", @"Grace", @"Linus"];
        for (NSString *name in names) {
            BLGreeter *greeter = [[BLGreeter alloc] initWithName:name];
            NSLog(@"%@", [greeter message]);
        }
    }
    return 0;
}
