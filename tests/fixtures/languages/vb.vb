Imports System
Imports System.Collections.Generic

Module BetlangFixture
    Sub Main(args As String())
        Dim names As New List(Of String) From {"Ada", "Grace", "Linus"}

        For Each name As String In names
            Console.WriteLine($"Hello {name}")
        Next
    End Sub

    Function Count(values As IEnumerable(Of String)) As Dictionary(Of String, Integer)
        Dim result As New Dictionary(Of String, Integer)
        For Each value As String In values
            If result.ContainsKey(value) Then
                result(value) += 1
            Else
                result(value) = 1
            End If
        Next
        Return result
    End Function
End Module
