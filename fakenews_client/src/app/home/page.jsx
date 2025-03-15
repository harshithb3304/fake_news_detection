"use client";

import React, { useState } from "react";
import { Calendar } from "@/components/ui/calendar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { format, parseISO } from "date-fns";
import { CalendarIcon, Newspaper } from "lucide-react";
import { cn } from "@/lib/utils";

export default function FakeNewsDetector() {
  const [title, setTitle] = useState("");
  const [date, setDate] = useState(null);
  const [content, setContent] = useState("");
  const [category, setCategory] = useState("");
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const categories = [
    "politicsNews", "Government News", "left-news", "politics",
    "worldnews", "News", "Middle-east", "US_News"
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    if (!title || !date || !content || !category) {
      alert("Please fill all the fields");
      return;
    }
  
    setIsLoading(true);
  
    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          content,  // Matches Flask "content" key
          category, // Matches Flask "category" key
          date: new Date(date).toISOString(),  // Ensures correct date format
        }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to analyze article");
      }
  
      const data = await response.json();
      console.log("Received data from Flask backend:", data);
      setResult(data);
    } catch (error) {
      console.error("Error analyzing article:", error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="container mx-auto py-10 px-4 max-w-4xl">
      <div className="flex flex-col items-center mb-10 text-center">
        <Newspaper className="h-12 w-12 text-primary mb-4" />
        <h1 className="text-3xl font-bold tracking-tight">Fake News Detector</h1>
        <p className="text-muted-foreground mt-2">
          Enter the details of a news article to check if it might be fake
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Article Analysis</CardTitle>
          <CardDescription>Provide the article details below for analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            
            {/* Title */}
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Article Title</label>
              <Input placeholder="Enter the title of the news article" value={title} onChange={(e) => setTitle(e.target.value)} />
            </div>

            {/* Date - Type & Select */}
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Publication Date</label>
              <div className="flex gap-2">
                {/* Date Picker */}
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" className={cn("flex-1 justify-start text-left font-normal", !date && "text-muted-foreground")}>
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {date ? format(parseISO(date), "PPP") : "Select date"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0">
                    <Calendar mode="single" selected={date ? parseISO(date) : null} onSelect={(d) => setDate(d ? d.toISOString().split("T")[0] : null)} initialFocus />
                  </PopoverContent>
                </Popover>
                
                {/* Manual Date Input */}
                <Input type="date" className="flex-1" value={date || ""} onChange={(e) => setDate(e.target.value)} />
              </div>
            </div>

            {/* Category Dropdown */}
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">News Category</label>
              <Select value={category} onValueChange={setCategory}>
                <SelectTrigger>
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  {categories.map((cat) => (
                    <SelectItem key={cat} value={cat}>{cat}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Article Content */}
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Article Content</label>
              <Textarea
                placeholder="Paste the article content here..."
                className="min-h-[200px]"
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
            </div>

            {/* Submit Button */}
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? "Analyzing..." : "Analyze Article"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Display Result */}
      {result && (
        <div className="mt-6 p-4 border rounded-lg">
          <h2 className="text-xl font-semibold">Analysis Result:</h2>
          {result.isFake ? (
            <p className="text-red-600 font-bold">⚠ This article is likely fake.</p>
          ) : (
            <p className="text-green-600 font-bold">✔ This article appears genuine.</p>
          )}
        </div>
      )}
    </div>
  );
}
