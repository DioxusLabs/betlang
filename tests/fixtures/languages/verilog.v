module counter (
    input wire clk,
    input wire rst,
    output reg [3:0] count
);

always @(posedge clk) begin
    if (rst) begin
        count <= 4'b0000;
    end else begin
        count <= count + 4'b0001;
    end
end

endmodule
