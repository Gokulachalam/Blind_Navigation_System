import java.util.*;
public class largest_ele_arr_nosort {
    
    public static void main(String[] args) {
        
        // Initialize an array
        int[] arr = {5, 10, 20, 15, 25, 30};
        Arrays.sort(arr);
        System.out.println(Arrays.toString(arr));
        
        // Set the first element as the maximum value
        int max = arr[0];
        
        // Loop through the array to find the maximum value
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }
        
        // Print the maximum value
        System.out.println("The largest element in the array is: " + max);
    }
}